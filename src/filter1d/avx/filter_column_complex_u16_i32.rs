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
use crate::sse::_shuffle;
use num_complex::Complex;
use num_traits::MulAdd;
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn _mm_addsub_epi32(a: __m128i, b: __m128i) -> __m128i {
    let c = _mm_sign_epi32(b, _mm_setr_epi32(-1, 1, -1, 1));
    _mm_add_epi32(a, c)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_addsub_epi32(a: __m256i, b: __m256i) -> __m256i {
    let c = _mm256_sign_epi32(b, _mm256_setr_epi32(-1, 1, -1, 1, -1, 1, -1, 1));
    _mm256_add_epi32(a, c)
}

#[inline(always)]
pub(crate) unsafe fn mq_complex_mla(
    acc_r: __m128i,
    r0i0: __m128i,
    r: __m128i,
    r_swop: __m128i,
) -> __m128i {
    let c0 = _mm_mullo_epi32(r0i0, r);
    let c1 = _mm_mullo_epi32(_mm_shuffle_epi32::<{ _shuffle(2, 3, 0, 1) }>(r0i0), r_swop);
    _mm_add_epi32(acc_r, _mm_addsub_epi32(c0, c1))
}

#[inline(always)]
pub(crate) unsafe fn mq_complex_mul(r0i0: __m128i, r: __m128i, r_swop: __m128i) -> __m128i {
    let c0 = _mm_mullo_epi32(r0i0, r);
    let c1 = _mm_mullo_epi32(_mm_shuffle_epi32::<{ _shuffle(2, 3, 0, 1) }>(r0i0), r_swop);
    _mm_addsub_epi32(c0, c1)
}

#[inline(always)]
pub(crate) unsafe fn mq256_complex_mla(
    acc: __m256i,
    r0i0: __m256i,
    r: __m256i,
    r_swop: __m256i,
    idx: __m256i,
) -> __m256i {
    let c0 = _mm256_mullo_epi32(r0i0, r);
    let c1 = _mm256_mullo_epi32(_mm256_permutevar8x32_epi32(r0i0, idx), r_swop);
    _mm256_add_epi32(acc, _mm256_addsub_epi32(c0, c1))
}

pub(crate) fn filter_avx_column_complex_u16_i32(
    arena: Arena,
    arena_src: &[&[Complex<i32>]],
    dst: &mut [u16],
    image_size: ImageSize,
    kernel: &[Complex<i16>],
) {
    unsafe {
        filter_avx_column_complex_u16_i32_impl(arena, arena_src, dst, image_size, kernel);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn filter_avx_column_complex_u16_i32_impl(
    arena: Arena,
    arena_src: &[&[Complex<i32>]],
    dst: &mut [u16],
    image_size: ImageSize,
    kernel: &[Complex<i16>],
) {
    unsafe {
        let full_width = image_size.width * arena.components;

        let length = kernel.len();

        let mut cx = 0usize;

        let o_kernel = kernel;

        let kernel = kernel
            .iter()
            .map(|x| Complex {
                re: x.re as i32,
                im: x.im as i32,
            })
            .collect::<Vec<Complex<i32>>>();

        let cwg = kernel.get_unchecked(0);
        let c_real = _mm256_set1_epi32(cwg.re);
        let c_img = _mm256_set1_epi32(cwg.im);

        let rnd = _mm256_set1_epi32(1 << 13);

        let v_idx = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);

        let pack_even = _mm256_setr_epi32(0, 2, 4, 6, -1, -1, -1, -1);

        while cx + 8 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = _mm256_loadu_si256(v_src.as_ptr().cast());
            let values1 = _mm256_loadu_si256(v_src.get_unchecked(4..).as_ptr().cast());

            let mut k0 = mq256_complex_mla(rnd, values0, c_real, c_img, v_idx);
            let mut k1 = mq256_complex_mla(rnd, values1, c_real, c_img, v_idx);

            for i in 1..length {
                let cwg = kernel.get_unchecked(i);
                let c_real = _mm256_set1_epi32(cwg.re);
                let c_img = _mm256_set1_epi32(cwg.im);

                let v0 = arena_src.get_unchecked(i);

                let values0 = _mm256_loadu_si256(v0.get_unchecked(cx..).as_ptr().cast());
                let values1 = _mm256_loadu_si256(v0.get_unchecked(cx + 4..).as_ptr().cast());

                k0 = mq256_complex_mla(k0, values0, c_real, c_img, v_idx);
                k1 = mq256_complex_mla(k1, values1, c_real, c_img, v_idx);
            }

            k0 = _mm256_srai_epi32::<14>(k0);
            k1 = _mm256_srai_epi32::<14>(k1);

            k0 = _mm256_permutevar8x32_epi32(k0, pack_even);
            k1 = _mm256_permutevar8x32_epi32(k1, pack_even);

            let z0 = _mm256_packus_epi32(k0, k1);
            _mm_storeu_si128(
                dst.get_unchecked_mut(cx..).as_mut_ptr().cast(),
                _mm256_castsi256_si128(z0),
            );
            cx += 8;
        }

        let cwg = kernel.get_unchecked(0);
        let c_real = _mm_set1_epi32(cwg.re);
        let c_img = _mm_set1_epi32(cwg.im);
        let rnd = _mm_set1_epi32(1 << 13);

        let shuf_table = _mm_set_epi8(29, 28, 25, 24, 21, 20, 17, 16, 13, 12, 9, 8, 5, 4, 1, 0);

        while cx + 4 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = _mm_loadu_si128(v_src.as_ptr().cast());
            let values1 = _mm_loadu_si128(v_src.get_unchecked(2..).as_ptr().cast());

            let mut k0 = _mm_add_epi32(rnd, mq_complex_mul(values0, c_real, c_img));
            let mut k1 = _mm_add_epi32(rnd, mq_complex_mul(values1, c_real, c_img));

            for i in 1..length {
                let cwg = kernel.get_unchecked(i);
                let c_real = _mm_set1_epi32(cwg.re);
                let c_img = _mm_set1_epi32(cwg.im);

                let v0 = arena_src.get_unchecked(i);

                let values0 = _mm_loadu_si128(v0.get_unchecked(cx..).as_ptr().cast());
                let values1 = _mm_loadu_si128(v0.get_unchecked(cx + 2..).as_ptr().cast());

                k0 = mq_complex_mla(k0, values0, c_real, c_img);
                k1 = mq_complex_mla(k1, values1, c_real, c_img);
            }

            k0 = _mm_srai_epi32::<14>(k0);
            k1 = _mm_srai_epi32::<14>(k1);

            let packed = _mm_shuffle_epi8(_mm_packus_epi32(k0, k1), shuf_table);

            _mm_storeu_si64(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), packed);
            cx += 4;
        }

        let coeff = *o_kernel.get_unchecked(0);

        for x in cx..full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(x..);

            let q_coeff = Complex::new(coeff.re as i32, coeff.im as i32);

            let mut k0 = q_coeff * wrap_complex::<i32>(v_src.get_unchecked(0));

            for i in 1..length {
                let coeff = *o_kernel.get_unchecked(i);
                let q_coeff = Complex::new(coeff.re, coeff.im);
                k0 = wrap_complex::<i32>(arena_src.get_unchecked(i).get_unchecked(x))
                    .mul_add(wrap_complex::<i16>(&q_coeff), k0);
            }

            *dst.get_unchecked_mut(x) = k0.re.to_c_approx_();
        }
    }
}
