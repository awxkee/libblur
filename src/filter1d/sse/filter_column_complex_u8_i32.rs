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
use crate::filter1d::filter_column_complex_q::wrap_complex;
use crate::filter1d::to_approx_storage_complex::ToApproxStorageComplex;
use crate::img_size::ImageSize;
use num_complex::Complex;
use num_traits::MulAdd;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn _mm_addsub_epi32(a: __m128i, b: __m128i) -> __m128i {
    let c = _mm_sign_epi32(b, _mm_setr_epi32(-1, 1, -1, 1));
    _mm_add_epi32(a, c)
}

#[inline(always)]
pub(crate) unsafe fn mq_complex_mla(
    acc: (__m128i, __m128i),
    r0i0: __m128i,
    r: __m128i,
    r_swop: __m128i,
) -> (__m128i, __m128i) {
    let xc00 = _mm_madd_epi16(r0i0, r);
    let xc10 = _mm_madd_epi16(r0i0, r_swop);

    let x_r = _mm_unpacklo_epi32(xc00, xc10);
    let x_c = _mm_unpackhi_epi32(xc00, xc10);

    (_mm_add_epi32(acc.0, x_r), _mm_add_epi32(acc.1, x_c))
}

pub(crate) fn filter_sse_column_complex_u8_i32(
    arena: Arena,
    arena_src: &[&[Complex<i16>]],
    dst: &mut [u8],
    image_size: ImageSize,
    kernel: &[Complex<i16>],
) {
    unsafe {
        filter_sse_column_complex_u8_i32_impl(arena, arena_src, dst, image_size, kernel);
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn filter_sse_column_complex_u8_i32_impl(
    arena: Arena,
    arena_src: &[&[Complex<i16>]],
    dst: &mut [u8],
    image_size: ImageSize,
    kernel: &[Complex<i16>],
) {
    unsafe {
        let full_width = image_size.width * arena.components;

        let length = kernel.len();

        let mut cx = 0usize;

        let o_kernel = kernel;

        let kernel: Vec<[i32; 2]> = kernel
            .iter()
            .map(|x| {
                let inv_i = (-x.im).to_ne_bytes();
                let i = x.im.to_ne_bytes();
                let r = x.re.to_ne_bytes();
                let c_r = i32::from_ne_bytes([r[0], r[1], inv_i[0], inv_i[1]]);
                let c_i = i32::from_ne_bytes([i[0], i[1], r[0], r[1]]);
                [c_r, c_i]
            })
            .collect::<Vec<[i32; 2]>>();

        let cwg = kernel.get_unchecked(0);
        let c_real = _mm_set1_epi32(cwg[0]);
        let c_img = _mm_set1_epi32(cwg[1]);

        let rnd = _mm_set1_epi32(1 << 14);

        let shuf_table = _mm_set_epi8(29, 28, 25, 24, 21, 20, 17, 16, 13, 12, 9, 8, 5, 4, 1, 0);

        while cx + 8 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = _mm_loadu_si128(v_src.as_ptr().cast());
            let values1 = _mm_loadu_si128(v_src.get_unchecked(4..).as_ptr().cast());

            let (mut k0, mut k1) = mq_complex_mla((rnd, rnd), values0, c_real, c_img);
            let (mut k2, mut k3) = mq_complex_mla((rnd, rnd), values1, c_real, c_img);

            for i in 1..length {
                let cwg = kernel.get_unchecked(i);
                let c_real = _mm_set1_epi32(cwg[0]);
                let c_img = _mm_set1_epi32(cwg[1]);

                let v0 = arena_src.get_unchecked(i);

                let values0 = _mm_loadu_si128(v0.get_unchecked(cx..).as_ptr().cast());
                let values1 = _mm_loadu_si128(v0.get_unchecked(cx + 4..).as_ptr().cast());

                (k0, k1) = mq_complex_mla((k0, k1), values0, c_real, c_img);
                (k2, k3) = mq_complex_mla((k2, k3), values1, c_real, c_img);
            }

            k0 = _mm_srai_epi32::<15>(k0);
            k1 = _mm_srai_epi32::<15>(k1);
            k2 = _mm_srai_epi32::<15>(k2);
            k3 = _mm_srai_epi32::<15>(k3);

            let packed = _mm_packus_epi16(
                _mm_shuffle_epi8(_mm_packus_epi32(k0, k1), shuf_table),
                _mm_shuffle_epi8(_mm_packus_epi32(k2, k3), shuf_table),
            );
            _mm_storeu_si64(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), packed);
            cx += 8;
        }

        while cx + 4 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = _mm_loadu_si128(v_src.as_ptr().cast());

            let (mut k0, mut k1) = mq_complex_mla((rnd, rnd), values0, c_real, c_img);

            for i in 1..length {
                let cwg = kernel.get_unchecked(i);
                let c_real = _mm_set1_epi32(cwg[0]);
                let c_img = _mm_set1_epi32(cwg[1]);

                let v0 = arena_src.get_unchecked(i);

                let values0 = _mm_loadu_si128(v0.get_unchecked(cx..).as_ptr().cast());

                (k0, k1) = mq_complex_mla((k0, k1), values0, c_real, c_img);
            }

            k0 = _mm_srai_epi32::<15>(k0);
            k1 = _mm_srai_epi32::<15>(k1);

            let packed = _mm_packus_epi16(
                _mm_shuffle_epi8(_mm_packus_epi32(k0, k1), shuf_table),
                _mm_setzero_si128(),
            );

            _mm_storeu_si32(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), packed);
            cx += 4;
        }

        let coeff = *o_kernel.get_unchecked(0);

        for x in cx..full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(x..);

            let q_coeff = Complex::new(coeff.re as i32, coeff.im as i32);

            let mut k0 = q_coeff * wrap_complex::<i16>(v_src.get_unchecked(0));

            for i in 1..length {
                let coeff = *o_kernel.get_unchecked(i);
                let q_coeff = Complex::new(coeff.re, coeff.im);
                k0 = wrap_complex::<i16>(arena_src.get_unchecked(i).get_unchecked(x))
                    .mul_add(wrap_complex::<i16>(&q_coeff), k0);
            }

            *dst.get_unchecked_mut(x) = k0.re.to_c_approx_();
        }
    }
}
