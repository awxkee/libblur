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
use crate::avx::_mm256_packus_four_epi32;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn _mm256_mul_epi8_by_ps_x4<const FMA: bool>(
    input: __m256i,
    weight: __m256,
) -> (__m256, __m256, __m256, __m256) {
    let lo_16 = _mm256_unpacklo_epi8(input, _mm256_setzero_si256());
    let hi_16 = _mm256_unpackhi_epi8(input, _mm256_setzero_si256());

    let j0 = _mm256_unpacklo_epi16(lo_16, _mm256_setzero_si256());
    let j1 = _mm256_unpackhi_epi16(lo_16, _mm256_setzero_si256());
    let j2 = _mm256_unpacklo_epi16(hi_16, _mm256_setzero_si256());
    let j3 = _mm256_unpackhi_epi16(hi_16, _mm256_setzero_si256());

    let o0 = _mm256_cvtepi32_ps(j0);
    let o1 = _mm256_cvtepi32_ps(j1);
    let o2 = _mm256_cvtepi32_ps(j2);
    let o3 = _mm256_cvtepi32_ps(j3);

    (
        _mm256_mul_ps(o0, weight),
        _mm256_mul_ps(o1, weight),
        _mm256_mul_ps(o2, weight),
        _mm256_mul_ps(o3, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm256_mul_epi16_by_ps_x2<const FMA: bool>(
    input: __m256i,
    weight: __m256,
) -> (__m256, __m256) {
    let j0 = _mm256_unpacklo_epi16(input, _mm256_setzero_si256());
    let j1 = _mm256_unpackhi_epi16(input, _mm256_setzero_si256());

    let o0 = _mm256_cvtepi32_ps(j0);
    let o1 = _mm256_cvtepi32_ps(j1);

    (_mm256_mul_ps(o0, weight), _mm256_mul_ps(o1, weight))
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
pub(crate) unsafe fn _mm256_mul_epu16_widen(input: __m256i, weight: __m256i) -> (__m256i, __m256i) {
    let v_lo = _mm256_unpacklo_epi16(input, _mm256_setzero_si256());
    let v_hi = _mm256_unpackhi_epi16(input, _mm256_setzero_si256());
    (
        _mm256_mullo_epi32(v_lo, weight),
        _mm256_mullo_epi32(v_hi, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm256_mul_add_epi8_by_epi16_x4(
    accumulator: (__m256i, __m256i),
    input: __m256i,
    weight: __m256i,
) -> (__m256i, __m256i) {
    let j0 = _mm256_unpacklo_epi8(input, input);
    let j1 = _mm256_unpackhi_epi8(input, input);

    let vj0 = _mm256_mulhrs_epi16(_mm256_srli_epi16::<2>(j0), weight);
    let vj1 = _mm256_mulhrs_epi16(_mm256_srli_epi16::<2>(j1), weight);
    (
        _mm256_add_epi16(accumulator.0, vj0),
        _mm256_add_epi16(accumulator.1, vj1),
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
pub(crate) unsafe fn _mm256_mul_add_symm_epu16_by_epu16_x4(
    accumulator: (__m256i, __m256i),
    input0: __m256i,
    input1: __m256i,
    weight: __m256i,
) -> (__m256i, __m256i) {
    let a0 = _mm256_unpacklo_epi16(input0, _mm256_setzero_si256());
    let a1 = _mm256_unpacklo_epi16(input1, _mm256_setzero_si256());
    let a2 = _mm256_unpackhi_epi16(input0, _mm256_setzero_si256());
    let a3 = _mm256_unpackhi_epi16(input1, _mm256_setzero_si256());

    let lo_16 = _mm256_add_epi32(a0, a1);
    let hi_16 = _mm256_add_epi32(a2, a3);

    let vj0 = _mm256_mullo_epi32(lo_16, weight);
    let vj1 = _mm256_mullo_epi32(hi_16, weight);

    (
        _mm256_add_epi32(accumulator.0, vj0),
        _mm256_add_epi32(accumulator.1, vj1),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm256_opt_fmlaf_ps<const FMA: bool>(
    a: __m256,
    b: __m256,
    c: __m256,
) -> __m256 {
    if FMA {
        _mm256_fmadd_ps(b, c, a)
    } else {
        _mm256_add_ps(_mm256_mul_ps(b, c), a)
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm256_mul_add_epi8_by_ps_x4<const FMA: bool>(
    accumulator: (__m256, __m256, __m256, __m256),
    input: __m256i,
    weight: __m256,
) -> (__m256, __m256, __m256, __m256) {
    let lo_16 = _mm256_unpacklo_epi8(input, _mm256_setzero_si256());
    let hi_16 = _mm256_unpackhi_epi8(input, _mm256_setzero_si256());

    let j0 = _mm256_unpacklo_epi16(lo_16, _mm256_setzero_si256());
    let j1 = _mm256_unpackhi_epi16(lo_16, _mm256_setzero_si256());
    let j2 = _mm256_unpacklo_epi16(hi_16, _mm256_setzero_si256());
    let j3 = _mm256_unpackhi_epi16(hi_16, _mm256_setzero_si256());

    let a0 = _mm256_cvtepi32_ps(j0);
    let a1 = _mm256_cvtepi32_ps(j1);
    let a2 = _mm256_cvtepi32_ps(j2);
    let a3 = _mm256_cvtepi32_ps(j3);

    (
        _mm256_opt_fmlaf_ps::<FMA>(accumulator.0, a0, weight),
        _mm256_opt_fmlaf_ps::<FMA>(accumulator.1, a1, weight),
        _mm256_opt_fmlaf_ps::<FMA>(accumulator.2, a2, weight),
        _mm256_opt_fmlaf_ps::<FMA>(accumulator.3, a3, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm256_mul_add_symm_epi8_by_ps_x4<const FMA: bool>(
    accumulator: (__m256, __m256, __m256, __m256),
    input0: __m256i,
    input1: __m256i,
    weight: __m256,
) -> (__m256, __m256, __m256, __m256) {
    let j0 = _mm256_unpacklo_epi8(input0, _mm256_setzero_si256());
    let j1 = _mm256_unpacklo_epi8(input1, _mm256_setzero_si256());
    let j2 = _mm256_unpackhi_epi8(input0, _mm256_setzero_si256());
    let j3 = _mm256_unpackhi_epi8(input1, _mm256_setzero_si256());

    let lo_16 = _mm256_add_epi16(j0, j1);
    let hi_16 = _mm256_add_epi16(j2, j3);

    let a0 = _mm256_unpacklo_epi16(lo_16, _mm256_setzero_si256());
    let a1 = _mm256_unpackhi_epi16(lo_16, _mm256_setzero_si256());
    let a2 = _mm256_unpacklo_epi16(hi_16, _mm256_setzero_si256());
    let a3 = _mm256_unpackhi_epi16(hi_16, _mm256_setzero_si256());

    let v0 = _mm256_cvtepi32_ps(a0);
    let v1 = _mm256_cvtepi32_ps(a1);
    let v2 = _mm256_cvtepi32_ps(a2);
    let v3 = _mm256_cvtepi32_ps(a3);

    (
        _mm256_opt_fmlaf_ps::<FMA>(accumulator.0, v0, weight),
        _mm256_opt_fmlaf_ps::<FMA>(accumulator.1, v1, weight),
        _mm256_opt_fmlaf_ps::<FMA>(accumulator.2, v2, weight),
        _mm256_opt_fmlaf_ps::<FMA>(accumulator.3, v3, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm256_mul_add_symm_epi16_by_ps_x2<const FMA: bool>(
    accumulator: (__m256, __m256),
    input0: __m256i,
    input1: __m256i,
    weight: __m256,
) -> (__m256, __m256) {
    let j0 = _mm256_unpacklo_epi16(input0, _mm256_setzero_si256());
    let j1 = _mm256_unpacklo_epi16(input1, _mm256_setzero_si256());
    let j2 = _mm256_unpackhi_epi16(input0, _mm256_setzero_si256());
    let j3 = _mm256_unpackhi_epi16(input1, _mm256_setzero_si256());

    let a0 = _mm256_add_epi32(j0, j1);
    let a1 = _mm256_add_epi32(j2, j3);

    let v0 = _mm256_cvtepi32_ps(a0);
    let v1 = _mm256_cvtepi32_ps(a1);

    (
        _mm256_opt_fmlaf_ps::<FMA>(accumulator.0, v0, weight),
        _mm256_opt_fmlaf_ps::<FMA>(accumulator.1, v1, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm256_pack_ps_x4_epi8(store: (__m256, __m256, __m256, __m256)) -> __m256i {
    let v0 = _mm256_cvtps_epi32(store.0);
    let v1 = _mm256_cvtps_epi32(store.1);
    let v2 = _mm256_cvtps_epi32(store.2);
    let v3 = _mm256_cvtps_epi32(store.3);

    _mm256_packus_four_epi32(v0, v1, v2, v3)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_pack_ps_x2_epi16(store: (__m256, __m256)) -> __m256i {
    let v0 = _mm256_cvtps_epi32(store.0);
    let v1 = _mm256_cvtps_epi32(store.1);
    _mm256_packus_epi32(v0, v1)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_pack_epi32_x4_epi8(store: (__m256i, __m256i)) -> __m256i {
    let rnd = _mm256_set1_epi16((1 << 5) - 1);
    _mm256_packus_epi16(
        _mm256_srai_epi16::<6>(_mm256_add_epi16(store.0, rnd)),
        _mm256_srai_epi16::<6>(_mm256_add_epi16(store.1, rnd)),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm256_pack_epi32_x4_epu16(store: (__m256i, __m256i)) -> __m256i {
    let rnd = _mm256_set1_epi32((1 << 14) - 1);
    _mm256_packus_epi32(
        _mm256_srli_epi32::<15>(_mm256_add_epi32(store.0, rnd)),
        _mm256_srli_epi32::<15>(_mm256_add_epi32(store.1, rnd)),
    )
}
