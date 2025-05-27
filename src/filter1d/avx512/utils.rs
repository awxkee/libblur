/*
 * // Copyright (c) Radzivon Bartoshyk 5/2025. All rights reserved.
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
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn _mm_fmlaf_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    _mm_fmadd_ps(b, c, a)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_fmlaf_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_fmadd_ps(b, c, a)
}

#[inline(always)]
pub(crate) unsafe fn _mm512_fmlaf_ps(a: __m512, b: __m512, c: __m512) -> __m512 {
    _mm512_fmadd_ps(b, c, a)
}

#[inline(always)]
pub(crate) unsafe fn _mm512_fmla_pd(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    _mm512_fmadd_pd(b, c, a)
}

#[inline(always)]
pub(crate) unsafe fn _mm512_mul_epu16_widen(input: __m512i, weight: __m512i) -> (__m512i, __m512i) {
    let v_lo = _mm512_unpacklo_epi16(input, _mm512_setzero_si512());
    let v_hi = _mm512_unpackhi_epi16(input, _mm512_setzero_si512());
    (
        _mm512_mullo_epi32(v_lo, weight),
        _mm512_mullo_epi32(v_hi, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm512_mul_add_symm_epu16_by_epu16_x4(
    accumulator: (__m512i, __m512i),
    input0: __m512i,
    input1: __m512i,
    weight: __m512i,
) -> (__m512i, __m512i) {
    let a0 = _mm512_unpacklo_epi16(input0, _mm512_setzero_si512());
    let a1 = _mm512_unpacklo_epi16(input1, _mm512_setzero_si512());
    let a2 = _mm512_unpackhi_epi16(input0, _mm512_setzero_si512());
    let a3 = _mm512_unpackhi_epi16(input1, _mm512_setzero_si512());

    let lo_16 = _mm512_add_epi32(a0, a1);
    let hi_16 = _mm512_add_epi32(a2, a3);

    let vj0 = _mm512_mullo_epi32(lo_16, weight);
    let vj1 = _mm512_mullo_epi32(hi_16, weight);

    (
        _mm512_add_epi32(accumulator.0, vj0),
        _mm512_add_epi32(accumulator.1, vj1),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm512_pack_epi32_x4_epu16(store: (__m512i, __m512i)) -> __m512i {
    let rnd = _mm512_set1_epi32((1 << 14) - 1);
    _mm512_packus_epi32(
        _mm512_srli_epi32::<15>(_mm512_add_epi32(store.0, rnd)),
        _mm512_srli_epi32::<15>(_mm512_add_epi32(store.1, rnd)),
    )
}

#[inline(always)]
pub(crate) const fn v512_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
pub(crate) unsafe fn _mm256_mul_epi8_by_epi16_x4(
    input: __m256i,
    weight: __m256i,
) -> (__m256i, __m256i) {
    let j0 = _mm256_unpacklo_epi8(input, input);
    let j1 = _mm256_unpackhi_epi8(input, input);

    (
        _mm256_mulhrs_epi16(_mm256_srli_epi16::<2>(j0), weight),
        _mm256_mulhrs_epi16(_mm256_srli_epi16::<2>(j1), weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm512_mul_epi8_by_epi16_x4(
    input: __m512i,
    weight: __m512i,
) -> (__m512i, __m512i) {
    let j0 = _mm512_unpacklo_epi8(input, input);
    let j1 = _mm512_unpackhi_epi8(input, input);

    (
        _mm512_mulhrs_epi16(_mm512_srli_epi16::<2>(j0), weight),
        _mm512_mulhrs_epi16(_mm512_srli_epi16::<2>(j1), weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm256_mul_add_symm_epi8_by_epi16_x4(
    accumulator: (__m256i, __m256i),
    input0: __m256i,
    input1: __m256i,
    weight: __m256i,
) -> (__m256i, __m256i) {
    let a0 = _mm256_unpacklo_epi8(input0, _mm256_setzero_si256());
    let a1 = _mm256_unpacklo_epi8(input1, _mm256_setzero_si256());
    let a2 = _mm256_unpackhi_epi8(input0, _mm256_setzero_si256());
    let a3 = _mm256_unpackhi_epi8(input1, _mm256_setzero_si256());

    let lo_16 = _mm256_add_epi16(a0, a1);
    let hi_16 = _mm256_add_epi16(a2, a3);

    let vj0 = _mm256_mulhrs_epi16(_mm256_slli_epi16::<6>(lo_16), weight);
    let vj1 = _mm256_mulhrs_epi16(_mm256_slli_epi16::<6>(hi_16), weight);

    (
        _mm256_add_epi16(accumulator.0, vj0),
        _mm256_add_epi16(accumulator.1, vj1),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm512_mul_add_symm_epi8_by_epi16_x4(
    accumulator: (__m512i, __m512i),
    input0: __m512i,
    input1: __m512i,
    weight: __m512i,
) -> (__m512i, __m512i) {
    let a0 = _mm512_unpacklo_epi8(input0, _mm512_setzero_si512());
    let a1 = _mm512_unpacklo_epi8(input1, _mm512_setzero_si512());
    let a2 = _mm512_unpackhi_epi8(input0, _mm512_setzero_si512());
    let a3 = _mm512_unpackhi_epi8(input1, _mm512_setzero_si512());

    let lo_16 = _mm512_add_epi16(a0, a1);
    let hi_16 = _mm512_add_epi16(a2, a3);

    let vj0 = _mm512_mulhrs_epi16(_mm512_slli_epi16::<6>(lo_16), weight);
    let vj1 = _mm512_mulhrs_epi16(_mm512_slli_epi16::<6>(hi_16), weight);

    (
        _mm512_add_epi16(accumulator.0, vj0),
        _mm512_add_epi16(accumulator.1, vj1),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn _mm256_pack_epi32_x4_epi8(store: (__m256i, __m256i)) -> __m256i {
    let rnd = _mm256_set1_epi16((1 << 5) - 1);
    _mm256_packus_epi16(
        _mm256_srai_epi16::<6>(_mm256_add_epi16(store.0, rnd)),
        _mm256_srai_epi16::<6>(_mm256_add_epi16(store.1, rnd)),
    )
}

#[inline]
#[target_feature(enable = "avx512bw")]
pub(crate) unsafe fn _mm512_pack_epi32_x4_epi8(store: (__m512i, __m512i)) -> __m512i {
    let rnd = _mm512_set1_epi16((1 << 5) - 1);
    _mm512_packus_epi16(
        _mm512_srai_epi16::<6>(_mm512_add_epi16(store.0, rnd)),
        _mm512_srai_epi16::<6>(_mm512_add_epi16(store.1, rnd)),
    )
}
