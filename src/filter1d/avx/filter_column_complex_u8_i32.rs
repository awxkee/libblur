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
use crate::avx::shuffle;
use crate::filter1d::arena::Arena;
use crate::filter1d::filter_column_complex_q::wrap_complex;
use crate::filter1d::to_approx_storage_complex::ToApproxStorageComplex;
use crate::img_size::ImageSize;
use num_complex::Complex;
use num_traits::MulAdd;
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn v_interleave_i32x8(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    let xy_l = _mm256_unpacklo_epi32(a, b);
    let xy_h = _mm256_unpackhi_epi32(a, b);

    let xy0 = _mm256_permute2x128_si256::<32>(xy_l, xy_h);
    let xy1 = _mm256_permute2x128_si256::<49>(xy_l, xy_h);
    (xy0, xy1)
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

#[inline(always)]
pub(crate) unsafe fn mq256_complex_mla(
    acc: (__m256i, __m256i),
    r0i0: __m256i,
    r: __m256i,
    r_swop: __m256i,
) -> (__m256i, __m256i) {
    let x_r = _mm256_madd_epi16(r0i0, r);
    let x_c = _mm256_madd_epi16(r0i0, r_swop);

    let (z_r, z_c) = v_interleave_i32x8(x_r, x_c);

    (_mm256_add_epi32(acc.0, z_r), _mm256_add_epi32(acc.1, z_c))
}

pub(crate) fn filter_avx_column_complex_u8_i32(
    arena: Arena,
    arena_src: &[&[Complex<i16>]],
    dst: &mut [u8],
    image_size: ImageSize,
    kernel: &[Complex<i16>],
) {
    unsafe {
        filter_avx_column_complex_u8_i32_impl(arena, arena_src, dst, image_size, kernel);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn filter_avx_column_complex_u8_i32_impl(
    arena: Arena,
    arena_src: &[&[Complex<i16>]],
    dst: &mut [u8],
    image_size: ImageSize,
    kernel: &[Complex<i16>],
) {
    unsafe {
        let full_width = image_size.width * arena.components;

        let length = kernel.len();

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

        let mut cx = 0usize;

        let cwg = kernel.get_unchecked(0);
        let c_real = _mm256_set1_epi32(cwg[0]);
        let c_img = _mm256_set1_epi32(cwg[1]);

        let rnd = _mm256_set1_epi32(1 << 14);

        let shuf_table = _mm256_set_epi8(
            29, 28, 25, 24, 21, 20, 17, 16, 13, 12, 9, 8, 5, 4, 1, 0, 29, 28, 25, 24, 21, 20, 17,
            16, 13, 12, 9, 8, 5, 4, 1, 0,
        );

        while cx + 16 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = _mm256_loadu_si256(v_src.as_ptr().cast());
            let values1 = _mm256_loadu_si256(v_src.get_unchecked(8..).as_ptr().cast());

            let (mut k0, mut k1) = mq256_complex_mla((rnd, rnd), values0, c_real, c_img);
            let (mut k2, mut k3) = mq256_complex_mla((rnd, rnd), values1, c_real, c_img);

            for i in 1..length {
                let cwg = kernel.get_unchecked(i);
                let c_real = _mm256_set1_epi32(cwg[0]);
                let c_img = _mm256_set1_epi32(cwg[1]);

                let v0 = arena_src.get_unchecked(i);

                let values0 = _mm256_loadu_si256(v0.get_unchecked(cx..).as_ptr().cast());
                let values1 = _mm256_loadu_si256(v0.get_unchecked(cx + 8..).as_ptr().cast());

                (k0, k1) = mq256_complex_mla((k0, k1), values0, c_real, c_img);
                (k2, k3) = mq256_complex_mla((k2, k3), values1, c_real, c_img);
            }

            k0 = _mm256_srai_epi32::<15>(k0);
            k1 = _mm256_srai_epi32::<15>(k1);
            k2 = _mm256_srai_epi32::<15>(k2);
            k3 = _mm256_srai_epi32::<15>(k3);

            const M: i32 = shuffle(3, 1, 2, 0);
            let z0 = _mm256_shuffle_epi8(
                _mm256_permute4x64_epi64::<M>(_mm256_packus_epi32(k0, k1)),
                shuf_table,
            );
            let z1 = _mm256_shuffle_epi8(
                _mm256_permute4x64_epi64::<M>(_mm256_packus_epi32(k2, k3)),
                shuf_table,
            );

            let c0 = _mm256_castsi256_si128(z0);
            let c1 = _mm256_castsi256_si128(z1);
            let packed = _mm_packus_epi16(c0, c1);
            _mm_storeu_si128(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), packed);
            cx += 16;
        }

        while cx + 8 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = _mm256_loadu_si256(v_src.as_ptr().cast());

            let (mut k0, mut k1) = mq256_complex_mla((rnd, rnd), values0, c_real, c_img);

            for i in 1..length {
                let cwg = kernel.get_unchecked(i);
                let c_real = _mm256_set1_epi32(cwg[0]);
                let c_img = _mm256_set1_epi32(cwg[1]);

                let v0 = arena_src.get_unchecked(i);

                let values0 = _mm256_loadu_si256(v0.get_unchecked(cx..).as_ptr().cast());

                (k0, k1) = mq256_complex_mla((k0, k1), values0, c_real, c_img);
            }

            k0 = _mm256_srai_epi32::<15>(k0);
            k1 = _mm256_srai_epi32::<15>(k1);

            const M: i32 = shuffle(3, 1, 2, 0);
            let z0 = _mm256_shuffle_epi8(
                _mm256_permute4x64_epi64::<M>(_mm256_packus_epi32(k0, k1)),
                shuf_table,
            );

            let c0 = _mm256_castsi256_si128(z0);
            let packed = _mm_packus_epi16(c0, c0);
            _mm_storeu_si64(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), packed);
            cx += 8;
        }

        let rnd = _mm256_castsi256_si128(rnd);
        let c_real = _mm256_castsi256_si128(c_real);
        let c_img = _mm256_castsi256_si128(c_img);
        let shuf_table = _mm256_castsi256_si128(shuf_table);

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
