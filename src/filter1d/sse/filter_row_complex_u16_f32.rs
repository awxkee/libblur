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

pub(crate) fn filter_sse_row_complex_u16_f32(
    arena: Arena,
    arena_src: &[u16],
    dst: &mut [Complex<f32>],
    image_size: ImageSize,
    kernel: &[Complex<f32>],
) {
    unsafe {
        filter_row_complex_u16_f32_impl(arena, arena_src, dst, image_size, kernel);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn filter_row_complex_u16_f32_impl(
    arena: Arena,
    arena_src: &[u16],
    dst: &mut [Complex<f32>],
    image_size: ImageSize,
    kernel: &[Complex<f32>],
) {
    unsafe {
        let width = image_size.width;

        let src = arena_src;

        let local_src = src;

        let length = kernel.len();

        let mut cx = 0usize;

        let max_width = width * arena.components;

        let c_re = _mm_set1_ps(kernel.get_unchecked(0).re);
        let c_im = _mm_set1_ps(kernel.get_unchecked(0).im);

        while cx + 8 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let values = _mm_loadu_si128(shifted_src.as_ptr().cast());

            let a0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(values, _mm_setzero_si128()));
            let a1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(values, _mm_setzero_si128()));

            let mut r0 = _mm_mul_ps(a0, c_re);
            let mut r1 = _mm_mul_ps(a1, c_re);

            let mut i0 = _mm_mul_ps(a0, c_im);
            let mut i1 = _mm_mul_ps(a1, c_im);

            for i in 1..length {
                let k = kernel.get_unchecked(i);
                let c_re = _mm_set1_ps(k.re);
                let c_im = _mm_set1_ps(k.im);

                let values = _mm_loadu_si128(
                    shifted_src
                        .get_unchecked(i * arena.components..)
                        .as_ptr()
                        .cast(),
                );

                let a0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(values, _mm_setzero_si128()));
                let a1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(values, _mm_setzero_si128()));

                r0 = _mm_add_ps(r0, _mm_mul_ps(a0, c_re));
                r1 = _mm_add_ps(r1, _mm_mul_ps(a1, c_re));

                i0 = _mm_add_ps(i0, _mm_mul_ps(a0, c_im));
                i1 = _mm_add_ps(i1, _mm_mul_ps(a1, c_im));
            }

            let dst0 = dst.get_unchecked_mut(cx..);

            let (z0, z1) = (_mm_unpacklo_ps(r0, i0), _mm_unpackhi_ps(r0, i0));
            let (z2, z3) = (_mm_unpacklo_ps(r1, i1), _mm_unpackhi_ps(r1, i1));

            _mm_storeu_ps(dst0.as_mut_ptr().cast(), z0);
            _mm_storeu_ps(dst0.get_unchecked_mut(2..).as_mut_ptr().cast(), z1);
            _mm_storeu_ps(dst0.get_unchecked_mut(4..).as_mut_ptr().cast(), z2);
            _mm_storeu_ps(dst0.get_unchecked_mut(6..).as_mut_ptr().cast(), z3);
            cx += 8;
        }

        while cx + 4 < max_width {
            let shifted_src = local_src.get_unchecked(cx..);

            let values = _mm_loadu_si64(shifted_src.as_ptr().cast());

            let a0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(values, _mm_setzero_si128()));

            let mut r0 = _mm_mul_ps(a0, c_re);
            let mut i0 = _mm_mul_ps(a0, c_im);

            for i in 1..length {
                let k = kernel.get_unchecked(i);
                let c_re = _mm_set1_ps(k.re);
                let c_im = _mm_set1_ps(k.im);

                let values = _mm_loadu_si64(
                    shifted_src
                        .get_unchecked(i * arena.components..)
                        .as_ptr()
                        .cast(),
                );

                let a0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(values, _mm_setzero_si128()));

                r0 = _mm_add_ps(r0, _mm_mul_ps(a0, c_re));
                i0 = _mm_add_ps(i0, _mm_mul_ps(a0, c_im));
            }

            let dst0 = dst.get_unchecked_mut(cx..);

            let (z0, z1) = (_mm_unpacklo_ps(r0, i0), _mm_unpackhi_ps(r0, i0));

            _mm_storeu_ps(dst0.as_mut_ptr().cast(), z0);
            _mm_storeu_ps(dst0.get_unchecked_mut(2..).as_mut_ptr().cast(), z1);
            cx += 4;
        }

        for x in cx..max_width {
            let shifted_src = local_src.get_unchecked(x..);
            let a0 = _mm_set1_ps(*shifted_src.get_unchecked(0) as f32);
            let mut r0 = _mm_mul_ps(a0, c_re);
            let mut i0 = _mm_mul_ps(a0, c_im);

            for i in 1..length {
                let k = kernel.get_unchecked(i);
                let c_re = _mm_set1_ps(k.re);
                let c_im = _mm_set1_ps(k.im);

                let a0 = _mm_set1_ps(*shifted_src.get_unchecked(i * arena.components) as f32);

                r0 = _mm_add_ps(r0, _mm_mul_ps(a0, c_re));
                i0 = _mm_add_ps(i0, _mm_mul_ps(a0, c_im));
            }
            _mm_store_ss(dst.get_unchecked_mut(x..).as_mut_ptr().cast(), r0);
            _mm_store_ss(
                (dst.get_unchecked_mut(x..).as_mut_ptr() as *mut f32).add(1),
                i0,
            );
        }
    }
}
