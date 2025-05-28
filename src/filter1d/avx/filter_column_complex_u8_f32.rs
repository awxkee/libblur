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
use crate::img_size::ImageSize;
use crate::to_storage::ToStorage;
use num_complex::Complex;
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn mq_complex_mla(
    acc_r: __m128,
    r0i0: __m128,
    r: __m128,
    r_swop: __m128,
) -> __m128 {
    let c0 = _mm_mul_ps(r0i0, r);
    let c1 = _mm_mul_ps(
        _mm_shuffle_ps::<{ shuffle(2, 3, 0, 1) }>(r0i0, r0i0),
        r_swop,
    );
    _mm_add_ps(acc_r, _mm_addsub_ps(c0, c1))
}

#[inline(always)]
pub(crate) unsafe fn _mm256_correct4x64_ps(a: __m256) -> __m256 {
    const M: i32 = shuffle(3, 1, 2, 0);
    _mm256_castsi256_ps(_mm256_permute4x64_epi64::<M>(_mm256_castps_si256(a)))
}

#[inline(always)]
pub(crate) unsafe fn mq256_complex_mla(
    acc_r: __m256,
    r0i0: __m256,
    r: __m256,
    r_swop: __m256,
) -> __m256 {
    let c0 = _mm256_mul_ps(r0i0, r);
    let c1 = _mm256_mul_ps(
        _mm256_correct4x64_ps(_mm256_shuffle_ps::<{ shuffle(2, 3, 0, 1) }>(r0i0, r0i0)),
        r_swop,
    );
    _mm256_add_ps(acc_r, _mm256_addsub_ps(c0, c1))
}

#[inline(always)]
pub(crate) unsafe fn mq256_complex_mul(r0i0: __m256, r: __m256, r_swop: __m256) -> __m256 {
    let c0 = _mm256_mul_ps(r0i0, r);
    let c1 = _mm256_mul_ps(
        _mm256_correct4x64_ps(_mm256_shuffle_ps::<{ shuffle(2, 3, 0, 1) }>(r0i0, r0i0)),
        r_swop,
    );
    _mm256_addsub_ps(c0, c1)
}

#[inline(always)]
pub(crate) unsafe fn mq_complex_mul(r0i0: __m128, r: __m128, r_swop: __m128) -> __m128 {
    let c0 = _mm_mul_ps(r0i0, r);
    let c1 = _mm_mul_ps(
        _mm_shuffle_ps::<{ shuffle(2, 3, 0, 1) }>(r0i0, r0i0),
        r_swop,
    );
    _mm_addsub_ps(c0, c1)
}

pub(crate) fn filter_avx_column_complex_u8_f32(
    arena: Arena,
    arena_src: &[&[Complex<f32>]],
    dst: &mut [u8],
    image_size: ImageSize,
    kernel: &[Complex<f32>],
) {
    unsafe {
        filter_avx_column_complex_u8_f32_impl(arena, arena_src, dst, image_size, kernel);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn filter_avx_column_complex_u8_f32_impl(
    arena: Arena,
    arena_src: &[&[Complex<f32>]],
    dst: &mut [u8],
    image_size: ImageSize,
    kernel: &[Complex<f32>],
) {
    unsafe {
        let full_width = image_size.width * arena.components;

        let length = kernel.len();

        let mut cx = 0usize;

        let cwg = _mm256_castsi256_ps(_mm256_set1_epi64x(
            (kernel.get_unchecked(0..).as_ptr() as *const i64).read_unaligned(),
        ));
        let c_real = _mm256_correct4x64_ps(_mm256_shuffle_ps::<0b10100000>(cwg, cwg));
        let c_img = _mm256_correct4x64_ps(_mm256_shuffle_ps::<0b11110101>(cwg, cwg));

        let shuf_table = _mm_set_epi8(29, 28, 25, 24, 21, 20, 17, 16, 13, 12, 9, 8, 5, 4, 1, 0);

        while cx + 16 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = _mm256_loadu_ps(v_src.as_ptr().cast());
            let values1 = _mm256_loadu_ps(v_src.get_unchecked(4..).as_ptr().cast());
            let values2 = _mm256_loadu_ps(v_src.get_unchecked(8..).as_ptr().cast());
            let values3 = _mm256_loadu_ps(v_src.get_unchecked(12..).as_ptr().cast());

            let mut k0 = mq256_complex_mul(values0, c_real, c_img);
            let mut k1 = mq256_complex_mul(values1, c_real, c_img);
            let mut k2 = mq256_complex_mul(values2, c_real, c_img);
            let mut k3 = mq256_complex_mul(values3, c_real, c_img);

            for i in 1..length {
                let cwg = _mm256_castsi256_ps(_mm256_set1_epi64x(
                    (kernel.get_unchecked(i..).as_ptr() as *const i64).read_unaligned(),
                ));
                let c_real = _mm256_correct4x64_ps(_mm256_shuffle_ps::<0b10100000>(cwg, cwg));
                let c_img = _mm256_correct4x64_ps(_mm256_shuffle_ps::<0b11110101>(cwg, cwg));

                let v0 = arena_src.get_unchecked(i);

                let values0 = _mm256_loadu_ps(v0.get_unchecked(cx..).as_ptr().cast());
                let values1 = _mm256_loadu_ps(v0.get_unchecked(cx + 4..).as_ptr().cast());
                let values2 = _mm256_loadu_ps(v0.get_unchecked(cx + 8..).as_ptr().cast());
                let values3 = _mm256_loadu_ps(v0.get_unchecked(cx + 12..).as_ptr().cast());

                k0 = mq256_complex_mla(k0, values0, c_real, c_img);
                k1 = mq256_complex_mla(k1, values1, c_real, c_img);
                k2 = mq256_complex_mla(k2, values2, c_real, c_img);
                k3 = mq256_complex_mla(k3, values3, c_real, c_img);
            }

            let v0 = _mm256_cvtps_epi32(k0);
            let v1 = _mm256_cvtps_epi32(k1);
            let v2 = _mm256_cvtps_epi32(k2);
            let v3 = _mm256_cvtps_epi32(k3);

            let s0 =
                _mm256_permute4x64_epi64::<{ shuffle(3, 1, 2, 0) }>(_mm256_packus_epi32(v0, v1));
            let s1 =
                _mm256_permute4x64_epi64::<{ shuffle(3, 1, 2, 0) }>(_mm256_packus_epi32(v2, v3));

            let packed0 = _mm_shuffle_epi8(_mm256_castsi256_si128(s0), shuf_table);
            let packed1 = _mm_shuffle_epi8(_mm256_extracti128_si256::<1>(s0), shuf_table);
            let packed2 = _mm_shuffle_epi8(_mm256_castsi256_si128(s1), shuf_table);
            let packed3 = _mm_shuffle_epi8(_mm256_extracti128_si256::<1>(s1), shuf_table);

            let packed01 = _mm_unpacklo_epi64(packed0, packed1);
            let packed23 = _mm_unpacklo_epi64(packed2, packed3);

            let packed0123 = _mm_packus_epi16(packed01, packed23);

            _mm_storeu_si128(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), packed0123);
            cx += 16;
        }

        let c_real = _mm256_castps256_ps128(c_real);
        let c_img = _mm256_castps256_ps128(c_img);

        while cx + 8 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = _mm_loadu_ps(v_src.as_ptr().cast());
            let values1 = _mm_loadu_ps(v_src.get_unchecked(2..).as_ptr().cast());
            let values2 = _mm_loadu_ps(v_src.get_unchecked(4..).as_ptr().cast());
            let values3 = _mm_loadu_ps(v_src.get_unchecked(6..).as_ptr().cast());

            let mut k0 = mq_complex_mul(values0, c_real, c_img);
            let mut k1 = mq_complex_mul(values1, c_real, c_img);
            let mut k2 = mq_complex_mul(values2, c_real, c_img);
            let mut k3 = mq_complex_mul(values3, c_real, c_img);

            for i in 1..length {
                let cwg = _mm_castsi128_ps(_mm_set1_epi64x(
                    (kernel.get_unchecked(i..).as_ptr() as *const i64).read_unaligned(),
                ));
                let c_real = _mm_shuffle_ps::<0b10100000>(cwg, cwg);
                let c_img = _mm_shuffle_ps::<0b11110101>(cwg, cwg);

                let v0 = arena_src.get_unchecked(i);

                let values0 = _mm_loadu_ps(v0.get_unchecked(cx..).as_ptr().cast());
                let values1 = _mm_loadu_ps(v0.get_unchecked(cx + 2..).as_ptr().cast());
                let values2 = _mm_loadu_ps(v0.get_unchecked(cx + 4..).as_ptr().cast());
                let values3 = _mm_loadu_ps(v0.get_unchecked(cx + 6..).as_ptr().cast());

                k0 = mq_complex_mla(k0, values0, c_real, c_img);
                k1 = mq_complex_mla(k1, values1, c_real, c_img);
                k2 = mq_complex_mla(k2, values2, c_real, c_img);
                k3 = mq_complex_mla(k3, values3, c_real, c_img);
            }

            let v0 = _mm_cvtps_epi32(k0);
            let v1 = _mm_cvtps_epi32(k1);
            let v2 = _mm_cvtps_epi32(k2);
            let v3 = _mm_cvtps_epi32(k3);

            let packed = _mm_packus_epi16(
                _mm_shuffle_epi8(_mm_packus_epi32(v0, v1), shuf_table),
                _mm_shuffle_epi8(_mm_packus_epi32(v2, v3), shuf_table),
            );
            _mm_storeu_si64(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), packed);
            cx += 8;
        }

        while cx + 4 < full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(cx..);

            let values0 = _mm_loadu_ps(v_src.as_ptr().cast());
            let values1 = _mm_loadu_ps(v_src.get_unchecked(2..).as_ptr().cast());

            let mut k0 = mq_complex_mul(values0, c_real, c_img);
            let mut k1 = mq_complex_mul(values1, c_real, c_img);

            for i in 1..length {
                let cwg = _mm_castsi128_ps(_mm_set1_epi64x(
                    (kernel.get_unchecked(i..).as_ptr() as *const i64).read_unaligned(),
                ));
                let c_real = _mm_shuffle_ps::<0b10100000>(cwg, cwg);
                let c_img = _mm_shuffle_ps::<0b11110101>(cwg, cwg);

                let v0 = arena_src.get_unchecked(i);

                let values0 = _mm_loadu_ps(v0.get_unchecked(cx..).as_ptr().cast());

                let values1 = _mm_loadu_ps(v0.get_unchecked(cx + 2..).as_ptr().cast());

                k0 = mq_complex_mla(k0, values0, c_real, c_img);
                k1 = mq_complex_mla(k1, values1, c_real, c_img);
            }

            let v0 = _mm_cvtps_epi32(k0);
            let v1 = _mm_cvtps_epi32(k1);

            let packed = _mm_packus_epi16(
                _mm_shuffle_epi8(_mm_packus_epi32(v0, v1), shuf_table),
                _mm_setzero_si128(),
            );

            _mm_storeu_si32(dst.get_unchecked_mut(cx..).as_mut_ptr().cast(), packed);
            cx += 4;
        }

        for x in cx..full_width {
            let v_src = arena_src.get_unchecked(0).get_unchecked(x..);

            let cwg = _mm_castsi128_ps(_mm_set1_epi64x(
                (v_src.as_ptr() as *const i64).read_unaligned(),
            ));

            let mut k0 = mq_complex_mul(cwg, c_real, c_img);

            for i in 1..length {
                let cwg = _mm_castsi128_ps(_mm_set1_epi64x(
                    (kernel.get_unchecked(i..).as_ptr() as *const i64).read_unaligned(),
                ));
                let c_real = cwg;
                let c_img = _mm_shuffle_ps::<0b10110001>(cwg, cwg);

                let ss = arena_src.get_unchecked(i).get_unchecked(x..);

                let vals = _mm_castsi128_ps(_mm_set1_epi64x(
                    (ss.as_ptr() as *const i64).read_unaligned(),
                ));

                k0 = mq_complex_mla(k0, vals, c_real, c_img);
            }

            *dst.get_unchecked_mut(x) = f32::from_bits(_mm_extract_ps::<0>(k0) as u32).to_();
        }
    }
}
