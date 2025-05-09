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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn _mm_mul_epi8_by_epi16_x4(
    input: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i) {
    let k0 = _mm_unpacklo_epi8(input, input);
    let k1 = _mm_unpackhi_epi8(input, input);

    (
        _mm_mulhrs_epi16(_mm_srli_epi16::<2>(k0), weight),
        _mm_mulhrs_epi16(_mm_srli_epi16::<2>(k1), weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_epu16_widen(input: __m128i, weight: __m128i) -> (__m128i, __m128i) {
    let k0 = _mm_unpacklo_epi16(input, _mm_setzero_si128());
    let k1 = _mm_unpackhi_epi16(input, _mm_setzero_si128());

    (_mm_mullo_epi32(k0, weight), _mm_mullo_epi32(k1, weight))
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_epi8_by_ps_x4<const FMA: bool>(
    input: __m128i,
    weight: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_unpacklo_epi8(input, zeros);
    let hi_16 = _mm_unpackhi_epi8(input, zeros);

    let a0 = _mm_unpacklo_epi16(lo_16, zeros);
    let a1 = _mm_unpackhi_epi16(lo_16, zeros);
    let a2 = _mm_unpacklo_epi16(hi_16, zeros);
    let a3 = _mm_unpackhi_epi16(hi_16, zeros);

    let v0 = _mm_cvtepi32_ps(a0);
    let v1 = _mm_cvtepi32_ps(a1);
    let v2 = _mm_cvtepi32_ps(a2);
    let v3 = _mm_cvtepi32_ps(a3);

    (
        _mm_mul_ps(v0, weight),
        _mm_mul_ps(v1, weight),
        _mm_mul_ps(v2, weight),
        _mm_mul_ps(v3, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_epi16_by_ps_x2<const FMA: bool>(
    input: __m128i,
    weight: __m128,
) -> (__m128, __m128) {
    let zeros = _mm_setzero_si128();
    let a0 = _mm_unpacklo_epi16(input, zeros);
    let a1 = _mm_unpackhi_epi16(input, zeros);

    let v0 = _mm_cvtepi32_ps(a0);
    let v1 = _mm_cvtepi32_ps(a1);

    (_mm_mul_ps(v0, weight), _mm_mul_ps(v1, weight))
}

#[inline(always)]
pub(crate) unsafe fn _mm_mull_epi8_by_epi16_x4(
    input: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i) {
    let k0 = _mm_unpacklo_epi8(input, input);
    let k1 = _mm_unpackhi_epi8(input, input);

    (
        _mm_mulhrs_epi16(_mm_srli_epi16::<2>(k0), weight),
        _mm_mulhrs_epi16(_mm_srli_epi16::<2>(k1), weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mull_epi8_by_epi16_x2(input: __m128i, weight: __m128i) -> __m128i {
    let lo_16 = _mm_unpacklo_epi8(input, input);

    _mm_mulhrs_epi16(_mm_srli_epi16::<2>(lo_16), weight)
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_epi8_by_ps_x2(input: __m128i, weight: __m128) -> (__m128, __m128) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_unpacklo_epi8(input, zeros);

    let k0 = _mm_unpacklo_epi16(lo_16, zeros);
    let k1 = _mm_unpackhi_epi16(lo_16, zeros);

    let a0 = _mm_cvtepi32_ps(k0);
    let a1 = _mm_cvtepi32_ps(k1);

    (_mm_mul_ps(a0, weight), _mm_mul_ps(a1, weight))
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_epi8_by_ps(input: __m128i, weight: __m128) -> __m128 {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_unpacklo_epi8(input, zeros);

    let k0 = _mm_unpacklo_epi16(lo_16, zeros);

    let a0 = _mm_cvtepi32_ps(k0);

    _mm_mul_ps(a0, weight)
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_epi16_by_ps<const FMA: bool>(
    input: __m128i,
    weight: __m128,
) -> __m128 {
    let zeros = _mm_setzero_si128();
    let k0 = _mm_unpacklo_epi16(input, zeros);
    let a0 = _mm_cvtepi32_ps(k0);

    _mm_mul_ps(a0, weight)
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_epi8_by_epi16_x2(input: __m128i, weight: __m128i) -> __m128i {
    let lo_16 = _mm_unpacklo_epi8(input, input);
    _mm_mulhrs_epi16(_mm_srli_epi16::<2>(lo_16), weight)
}

#[inline(always)]
pub(crate) unsafe fn _mm_opt_fmlaf_ps<const FMA: bool>(a: __m128, b: __m128, c: __m128) -> __m128 {
    if FMA {
        _mm_fmadd_ps(b, c, a)
    } else {
        _mm_add_ps(_mm_mul_ps(b, c), a)
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_opt_fmla_pd<const FMA: bool>(
    a: __m128d,
    b: __m128d,
    c: __m128d,
) -> __m128d {
    if FMA {
        _mm_fmadd_pd(b, c, a)
    } else {
        _mm_add_pd(_mm_mul_pd(b, c), a)
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_add_epi8_by_ps_x4<const FMA: bool>(
    accumulator: (__m128, __m128, __m128, __m128),
    input: __m128i,
    weight: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_unpacklo_epi8(input, zeros);
    let hi_16 = _mm_unpackhi_epi8(input, zeros);

    let k0 = _mm_unpacklo_epi16(lo_16, zeros);
    let k1 = _mm_unpackhi_epi16(lo_16, zeros);
    let k2 = _mm_unpacklo_epi16(hi_16, zeros);
    let k3 = _mm_unpackhi_epi16(hi_16, zeros);

    let a0 = _mm_cvtepi32_ps(k0);
    let a1 = _mm_cvtepi32_ps(k1);
    let a2 = _mm_cvtepi32_ps(k2);
    let a3 = _mm_cvtepi32_ps(k3);

    (
        _mm_opt_fmlaf_ps::<FMA>(accumulator.0, a0, weight),
        _mm_opt_fmlaf_ps::<FMA>(accumulator.1, a1, weight),
        _mm_opt_fmlaf_ps::<FMA>(accumulator.2, a2, weight),
        _mm_opt_fmlaf_ps::<FMA>(accumulator.3, a3, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mull_add_epi8_by_epi16_x4(
    accumulator: (__m128i, __m128i),
    input: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i) {
    let k0 = _mm_unpacklo_epi8(input, input);
    let k1 = _mm_unpackhi_epi8(input, input);

    let j0 = _mm_mulhrs_epi16(_mm_srli_epi16::<2>(k0), weight);
    let j1 = _mm_mulhrs_epi16(_mm_srli_epi16::<2>(k1), weight);
    (
        _mm_add_epi16(accumulator.0, j0),
        _mm_add_epi16(accumulator.1, j1),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mull_add_epi8_by_epi16_x2(
    accumulator: __m128i,
    input: __m128i,
    weight: __m128i,
) -> __m128i {
    let lo_16 = _mm_unpacklo_epi8(input, input);

    _mm_add_epi16(
        accumulator,
        _mm_mulhrs_epi16(_mm_srli_epi16::<2>(lo_16), weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mull_add_symm_epi8_by_epi16_x4(
    accumulator: (__m128i, __m128i),
    input0: __m128i,
    input1: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_add_epi16(
        _mm_unpacklo_epi8(input0, zeros),
        _mm_unpacklo_epi8(input1, zeros),
    );
    let hi_16 = _mm_add_epi16(
        _mm_unpackhi_epi8(input0, zeros),
        _mm_unpackhi_epi8(input1, zeros),
    );
    let j0 = _mm_mulhrs_epi16(_mm_slli_epi16::<6>(lo_16), weight);
    let j1 = _mm_mulhrs_epi16(_mm_slli_epi16::<6>(hi_16), weight);

    (
        _mm_add_epi16(accumulator.0, j0),
        _mm_add_epi16(accumulator.1, j1),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_madd_symm_epi8_by_epi16_x4(
    accumulator: (__m128i, __m128i),
    input0: __m128i,
    input1: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i) {
    let v0 = _mm_unpacklo_epi8(input0, _mm_setzero_si128());
    let v1 = _mm_unpackhi_epi8(input0, _mm_setzero_si128());
    let v2 = _mm_unpacklo_epi8(input1, _mm_setzero_si128());
    let v3 = _mm_unpackhi_epi8(input1, _mm_setzero_si128());

    let m0 = _mm_add_epi16(v0, v2);
    let m1 = _mm_add_epi16(v1, v3);

    let a0 = _mm_mullo_epi16(m0, weight);
    let a1 = _mm_mullo_epi16(m1, weight);

    (
        _mm_add_epi16(accumulator.0, a0),
        _mm_add_epi16(accumulator.1, a1),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_madd_s_epi8_by_epi16_x4(
    accumulator: (__m128i, __m128i),
    input0: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i) {
    let v0 = _mm_unpacklo_epi8(input0, _mm_setzero_si128());
    let v1 = _mm_unpackhi_epi8(input0, _mm_setzero_si128());

    let a0 = _mm_mullo_epi16(v0, weight);
    let a1 = _mm_mullo_epi16(v1, weight);

    (
        _mm_add_epi16(accumulator.0, a0),
        _mm_add_epi16(accumulator.1, a1),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_madd_epi8_by_epi16_x4(
    input0: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i) {
    let v0 = _mm_unpacklo_epi8(input0, _mm_setzero_si128());
    let v1 = _mm_unpackhi_epi8(input0, _mm_setzero_si128());

    let a0 = _mm_mullo_epi16(v0, weight);
    let a1 = _mm_mullo_epi16(v1, weight);

    (a0, a1)
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_add_symm_epi8_by_ps_x4<const FMA: bool>(
    accumulator: (__m128, __m128, __m128, __m128),
    input0: __m128i,
    input1: __m128i,
    weight: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_add_epi16(
        _mm_unpacklo_epi8(input0, zeros),
        _mm_unpacklo_epi8(input1, zeros),
    );
    let hi_16 = _mm_add_epi16(
        _mm_unpackhi_epi8(input0, zeros),
        _mm_unpackhi_epi8(input1, zeros),
    );

    let k0 = _mm_unpacklo_epi16(lo_16, zeros);
    let k1 = _mm_unpackhi_epi16(lo_16, zeros);
    let k2 = _mm_unpacklo_epi16(hi_16, zeros);
    let k3 = _mm_unpackhi_epi16(hi_16, zeros);

    let a0 = _mm_cvtepi32_ps(k0);
    let a1 = _mm_cvtepi32_ps(k1);
    let a2 = _mm_cvtepi32_ps(k2);
    let a3 = _mm_cvtepi32_ps(k3);

    (
        _mm_opt_fmlaf_ps::<FMA>(accumulator.0, a0, weight),
        _mm_opt_fmlaf_ps::<FMA>(accumulator.1, a1, weight),
        _mm_opt_fmlaf_ps::<FMA>(accumulator.2, a2, weight),
        _mm_opt_fmlaf_ps::<FMA>(accumulator.3, a3, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_add_symm_epi16_by_ps_x2<const FMA: bool>(
    accumulator: (__m128, __m128),
    input0: __m128i,
    input1: __m128i,
    weight: __m128,
) -> (__m128, __m128) {
    let zeros = _mm_setzero_si128();
    let l32 = _mm_add_epi32(
        _mm_unpacklo_epi16(input0, zeros),
        _mm_unpacklo_epi16(input1, zeros),
    );
    let h32 = _mm_add_epi32(
        _mm_unpackhi_epi16(input0, zeros),
        _mm_unpackhi_epi16(input1, zeros),
    );

    let a0 = _mm_cvtepi32_ps(l32);
    let a1 = _mm_cvtepi32_ps(h32);

    (
        _mm_opt_fmlaf_ps::<FMA>(accumulator.0, a0, weight),
        _mm_opt_fmlaf_ps::<FMA>(accumulator.1, a1, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_add_epi8_by_ps_x2<const FMA: bool>(
    accumulator: (__m128, __m128),
    input: __m128i,
    weight: __m128,
) -> (__m128, __m128) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_unpacklo_epi8(input, zeros);

    let k0 = _mm_unpacklo_epi16(lo_16, zeros);
    let k1 = _mm_unpackhi_epi16(lo_16, zeros);

    let a0 = _mm_cvtepi32_ps(k0);
    let a1 = _mm_cvtepi32_ps(k1);

    (
        _mm_opt_fmlaf_ps::<FMA>(accumulator.0, a0, weight),
        _mm_opt_fmlaf_ps::<FMA>(accumulator.1, a1, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_add_symm_epi8_by_ps_x2<const FMA: bool>(
    accumulator: (__m128, __m128),
    input0: __m128i,
    input1: __m128i,
    weight: __m128,
) -> (__m128, __m128) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_add_epi16(
        _mm_unpacklo_epi8(input0, zeros),
        _mm_unpacklo_epi8(input1, zeros),
    );

    let k0 = _mm_unpacklo_epi16(lo_16, zeros);
    let k1 = _mm_unpackhi_epi16(lo_16, zeros);

    let a0 = _mm_cvtepi32_ps(k0);
    let a1 = _mm_cvtepi32_ps(k1);

    (
        _mm_opt_fmlaf_ps::<FMA>(accumulator.0, a0, weight),
        _mm_opt_fmlaf_ps::<FMA>(accumulator.1, a1, weight),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_add_symm_epi8_by_ps<const FMA: bool>(
    accumulator: __m128,
    input0: __m128i,
    input1: __m128i,
    weight: __m128,
) -> __m128 {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_add_epi16(
        _mm_unpacklo_epi8(input0, zeros),
        _mm_unpacklo_epi8(input1, zeros),
    );

    let k0 = _mm_unpacklo_epi16(lo_16, zeros);

    let a0 = _mm_cvtepi32_ps(k0);

    _mm_opt_fmlaf_ps::<FMA>(accumulator, a0, weight)
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_add_symm_epi16_by_ps<const FMA: bool>(
    accumulator: __m128,
    input0: __m128i,
    input1: __m128i,
    weight: __m128,
) -> __m128 {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_add_epi32(
        _mm_unpacklo_epi16(input0, zeros),
        _mm_unpacklo_epi16(input1, zeros),
    );

    let a0 = _mm_cvtepi32_ps(lo_16);

    _mm_opt_fmlaf_ps::<FMA>(accumulator, a0, weight)
}

#[inline(always)]
pub(crate) unsafe fn _mm_mull_add_symm_epi8_by_epi16_x2(
    accumulator: __m128i,
    input0: __m128i,
    input1: __m128i,
    weight: __m128i,
) -> __m128i {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_add_epi16(
        _mm_unpacklo_epi8(input0, zeros),
        _mm_unpacklo_epi8(input1, zeros),
    );

    _mm_add_epi16(accumulator, _mm_mulhrs_epi16(lo_16, weight))
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_add_epi8_by_epi16_x4(
    accumulator: (__m128i, __m128i),
    input: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i) {
    let k0 = _mm_unpacklo_epi8(input, input);
    let k1 = _mm_unpackhi_epi8(input, input);

    let j0 = _mm_mulhrs_epi16(_mm_srli_epi16::<2>(k0), weight);
    let j1 = _mm_mulhrs_epi16(_mm_srli_epi16::<2>(k1), weight);

    (
        _mm_add_epi16(j0, accumulator.0),
        _mm_add_epi16(j1, accumulator.1),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_add_symm_epi8_by_epi16_x4(
    accumulator: (__m128i, __m128i),
    input0: __m128i,
    input1: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i) {
    let zeros = _mm_setzero_si128();
    let k0 = _mm_unpacklo_epi8(input0, zeros);
    let k1 = _mm_unpacklo_epi8(input1, zeros);
    let k2 = _mm_unpackhi_epi8(input0, zeros);
    let k3 = _mm_unpackhi_epi8(input1, zeros);
    let v0 = _mm_add_epi16(k0, k1);
    let v1 = _mm_add_epi16(k2, k3);

    let j0 = _mm_mulhrs_epi16(_mm_slli_epi16::<6>(v0), weight);
    let j1 = _mm_mulhrs_epi16(_mm_slli_epi16::<6>(v1), weight);

    (
        _mm_add_epi16(j0, accumulator.0),
        _mm_add_epi16(j1, accumulator.1),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_add_symm_epu16_by_epu16_x4(
    accumulator: (__m128i, __m128i),
    input0: __m128i,
    input1: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i) {
    let zeros = _mm_setzero_si128();
    let k0 = _mm_unpacklo_epi16(input0, zeros);
    let k1 = _mm_unpacklo_epi16(input1, zeros);
    let k2 = _mm_unpackhi_epi16(input0, zeros);
    let k3 = _mm_unpackhi_epi16(input1, zeros);
    let v0 = _mm_add_epi32(k0, k1);
    let v1 = _mm_add_epi32(k2, k3);

    let j0 = _mm_mullo_epi32(v0, weight);
    let j1 = _mm_mullo_epi32(v1, weight);

    (
        _mm_add_epi32(j0, accumulator.0),
        _mm_add_epi32(j1, accumulator.1),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_add_epi8_by_epi16_x2(
    accumulator: __m128i,
    input: __m128i,
    weight: __m128i,
) -> __m128i {
    let lo_16 = _mm_unpacklo_epi8(input, input);

    _mm_add_epi16(
        _mm_mulhrs_epi16(_mm_srli_epi16::<2>(lo_16), weight),
        accumulator,
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_add_symm_epi8_by_epi16_x2(
    accumulator: __m128i,
    input0: __m128i,
    input1: __m128i,
    weight: __m128i,
) -> __m128i {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_add_epi16(
        _mm_unpacklo_epi8(input0, zeros),
        _mm_unpacklo_epi8(input1, zeros),
    );

    _mm_add_epi16(
        _mm_mulhrs_epi16(_mm_slli_epi16::<6>(lo_16), weight),
        accumulator,
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_pack_epi32_x2_epi8(store: (__m128i, __m128i)) -> __m128i {
    let rnd = _mm_set1_epi16((1 << 5) - 1);
    _mm_packus_epi16(
        _mm_srai_epi16::<6>(_mm_add_epi16(store.0, rnd)),
        _mm_srai_epi16::<6>(_mm_add_epi16(store.1, rnd)),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_pack_epi32_x2_epu16(store: (__m128i, __m128i)) -> __m128i {
    let rnd = _mm_set1_epi32((1 << 14) - 1);
    _mm_packus_epi32(
        _mm_srli_epi32::<15>(_mm_add_epi32(store.0, rnd)),
        _mm_srli_epi32::<15>(_mm_add_epi32(store.1, rnd)),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_pack_epi32_epi8(store: __m128i) -> __m128i {
    let rnd = _mm_set1_epi16((1 << 5) - 1);
    _mm_packus_epi16(
        _mm_srai_epi16::<6>(_mm_add_epi16(store, rnd)),
        _mm_setzero_si128(),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_pack_ps_x4_epi8(store: (__m128, __m128, __m128, __m128)) -> __m128i {
    let o0 = _mm_cvtps_epi32(store.0);
    let o1 = _mm_cvtps_epi32(store.1);
    let o2 = _mm_cvtps_epi32(store.2);
    let o3 = _mm_cvtps_epi32(store.3);
    let hi_s = _mm_packs_epi32(o2, o3);
    let lo_s = _mm_packs_epi32(o0, o1);
    _mm_packus_epi16(lo_s, hi_s)
}

#[inline(always)]
pub(crate) unsafe fn _mm_pack_ps_x2_epi16(store: (__m128, __m128)) -> __m128i {
    let o0 = _mm_cvtps_epi32(store.0);
    let o1 = _mm_cvtps_epi32(store.1);
    _mm_packus_epi32(o0, o1)
}

#[inline(always)]
pub(crate) unsafe fn _mm_pack_ps_x2_epi8(store: (__m128, __m128)) -> __m128i {
    let lo_s = _mm_packs_epi32(_mm_cvtps_epi32(store.0), _mm_cvtps_epi32(store.1));
    _mm_packus_epi16(lo_s, _mm_setzero_si128())
}

#[inline(always)]
pub(crate) unsafe fn _mm_pack_ps_epi8(store: __m128) -> __m128i {
    let lo_s = _mm_packs_epi32(_mm_cvtps_epi32(store), _mm_setzero_si128());
    _mm_packus_epi16(lo_s, _mm_setzero_si128())
}

#[inline(always)]
pub(crate) unsafe fn _mm_pack_ps_epi16(store: __m128) -> __m128i {
    _mm_packus_epi32(_mm_cvtps_epi32(store), _mm_setzero_si128())
}

#[inline(always)]
pub(crate) unsafe fn _mm256_packus_four_epi32(
    a: __m256i,
    b: __m256i,
    c: __m256i,
    d: __m256i,
) -> __m256i {
    let ab = _mm256_packs_epi32(a, b);
    let cd = _mm256_packs_epi32(c, d);

    _mm256_packus_epi16(ab, cd)
}
