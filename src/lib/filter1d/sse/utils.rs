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

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_mul_epi8_by_epi16_x4(
    input: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_unpacklo_epi8(input, zeros);
    let hi_16 = _mm_unpackhi_epi8(input, zeros);

    (
        _mm_madd_epi16(_mm_unpacklo_epi16(lo_16, zeros), weight),
        _mm_madd_epi16(_mm_unpackhi_epi16(lo_16, zeros), weight),
        _mm_madd_epi16(_mm_unpacklo_epi16(hi_16, zeros), weight),
        _mm_madd_epi16(_mm_unpackhi_epi16(hi_16, zeros), weight),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_mul_epi8_by_ps_x4(
    input: __m128i,
    weight: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_unpacklo_epi8(input, zeros);
    let hi_16 = _mm_unpackhi_epi8(input, zeros);

    (
        _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(lo_16, zeros)), weight),
        _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(lo_16, zeros)), weight),
        _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(hi_16, zeros)), weight),
        _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(hi_16, zeros)), weight),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_mul_epi8_by_ps_x2(input: __m128i, weight: __m128) -> (__m128, __m128) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_unpacklo_epi8(input, zeros);

    (
        _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(lo_16, zeros)), weight),
        _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(lo_16, zeros)), weight),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_mul_epi8_by_epi16_x2(input: __m128i, weight: __m128i) -> (__m128i, __m128i) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_unpacklo_epi8(input, zeros);

    (
        _mm_madd_epi16(_mm_unpacklo_epi16(lo_16, zeros), weight),
        _mm_madd_epi16(_mm_unpackhi_epi16(lo_16, zeros), weight),
    )
}

#[inline]
#[target_feature(enable = "sse4.1,fma")]
pub unsafe fn _mm_fmlaf_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    _mm_fmadd_ps(b, c, a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_opt_fmlaf_ps<const FMA: bool>(a: __m128, b: __m128, c: __m128) -> __m128 {
    if FMA {
        _mm_fmlaf_ps(a, b, c)
    } else {
        _mm_add_ps(_mm_mul_ps(b, c), a)
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_mul_add_epi8_by_ps_x4<const FMA: bool>(
    accumulator: (__m128, __m128, __m128, __m128),
    input: __m128i,
    weight: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_unpacklo_epi8(input, zeros);
    let hi_16 = _mm_unpackhi_epi8(input, zeros);

    (
        _mm_opt_fmlaf_ps::<FMA>(
            accumulator.0,
            _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo_16, zeros)),
            weight,
        ),
        _mm_opt_fmlaf_ps::<FMA>(
            accumulator.1,
            _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo_16, zeros)),
            weight,
        ),
        _mm_opt_fmlaf_ps::<FMA>(
            accumulator.2,
            _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi_16, zeros)),
            weight,
        ),
        _mm_opt_fmlaf_ps::<FMA>(
            accumulator.3,
            _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi_16, zeros)),
            weight,
        ),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_mul_add_symm_epi8_by_ps_x4<const FMA: bool>(
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

    (
        _mm_opt_fmlaf_ps::<FMA>(
            accumulator.0,
            _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo_16, zeros)),
            weight,
        ),
        _mm_opt_fmlaf_ps::<FMA>(
            accumulator.1,
            _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo_16, zeros)),
            weight,
        ),
        _mm_opt_fmlaf_ps::<FMA>(
            accumulator.2,
            _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi_16, zeros)),
            weight,
        ),
        _mm_opt_fmlaf_ps::<FMA>(
            accumulator.3,
            _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi_16, zeros)),
            weight,
        ),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_mul_add_epi8_by_ps_x2<const FMA: bool>(
    accumulator: (__m128, __m128),
    input: __m128i,
    weight: __m128,
) -> (__m128, __m128) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_unpacklo_epi8(input, zeros);

    (
        _mm_opt_fmlaf_ps::<FMA>(
            accumulator.0,
            _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo_16, zeros)),
            weight,
        ),
        _mm_opt_fmlaf_ps::<FMA>(
            accumulator.1,
            _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo_16, zeros)),
            weight,
        ),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_mul_add_epi8_by_epi16_x4(
    accumulator: (__m128i, __m128i, __m128i, __m128i),
    input: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_unpacklo_epi8(input, zeros);
    let hi_16 = _mm_unpackhi_epi8(input, zeros);

    (
        _mm_add_epi32(
            _mm_madd_epi16(_mm_unpacklo_epi16(lo_16, zeros), weight),
            accumulator.0,
        ),
        _mm_add_epi32(
            _mm_madd_epi16(_mm_unpackhi_epi16(lo_16, zeros), weight),
            accumulator.1,
        ),
        _mm_add_epi32(
            _mm_madd_epi16(_mm_unpacklo_epi16(hi_16, zeros), weight),
            accumulator.2,
        ),
        _mm_add_epi32(
            _mm_madd_epi16(_mm_unpackhi_epi16(hi_16, zeros), weight),
            accumulator.3,
        ),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_mul_add_symm_epi8_by_epi16_x4(
    accumulator: (__m128i, __m128i, __m128i, __m128i),
    input0: __m128i,
    input1: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_add_epi16(
        _mm_unpacklo_epi8(input0, zeros),
        _mm_unpacklo_epi8(input1, zeros),
    );
    let hi_16 = _mm_add_epi16(
        _mm_unpackhi_epi8(input0, zeros),
        _mm_unpackhi_epi8(input1, zeros),
    );

    (
        _mm_add_epi32(
            _mm_madd_epi16(_mm_unpacklo_epi16(lo_16, zeros), weight),
            accumulator.0,
        ),
        _mm_add_epi32(
            _mm_madd_epi16(_mm_unpackhi_epi16(lo_16, zeros), weight),
            accumulator.1,
        ),
        _mm_add_epi32(
            _mm_madd_epi16(_mm_unpacklo_epi16(hi_16, zeros), weight),
            accumulator.2,
        ),
        _mm_add_epi32(
            _mm_madd_epi16(_mm_unpackhi_epi16(hi_16, zeros), weight),
            accumulator.3,
        ),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_mul_add_epi8_by_epi16_x2(
    accumulator: (__m128i, __m128i),
    input: __m128i,
    weight: __m128i,
) -> (__m128i, __m128i) {
    let zeros = _mm_setzero_si128();
    let lo_16 = _mm_unpacklo_epi8(input, zeros);

    (
        _mm_add_epi32(
            _mm_madd_epi16(_mm_unpacklo_epi16(lo_16, zeros), weight),
            accumulator.0,
        ),
        _mm_add_epi32(
            _mm_madd_epi16(_mm_unpackhi_epi16(lo_16, zeros), weight),
            accumulator.1,
        ),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_pack_epi32_x4_epi8(store: (__m128i, __m128i, __m128i, __m128i)) -> __m128i {
    let rounding_const = _mm_set1_epi32(1 << 14);
    let hi_s = _mm_packs_epi32(
        _mm_srai_epi32::<15>(_mm_add_epi32(store.2, rounding_const)),
        _mm_srai_epi32::<15>(_mm_add_epi32(store.3, rounding_const)),
    );
    let lo_s = _mm_packs_epi32(
        _mm_srai_epi32::<15>(_mm_add_epi32(store.0, rounding_const)),
        _mm_srai_epi32::<15>(_mm_add_epi32(store.1, rounding_const)),
    );
    _mm_packus_epi16(lo_s, hi_s)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_pack_epi32_x2_epi8(store: (__m128i, __m128i)) -> __m128i {
    let rounding_const = _mm_set1_epi32(1 << 14);
    let lo_s = _mm_packs_epi32(
        _mm_srai_epi32::<15>(_mm_add_epi32(store.0, rounding_const)),
        _mm_srai_epi32::<15>(_mm_add_epi32(store.1, rounding_const)),
    );
    _mm_packus_epi16(lo_s, _mm_setzero_si128())
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_pack_ps_x4_epi8(store: (__m128, __m128, __m128, __m128)) -> __m128i {
    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    let hi_s = _mm_packs_epi32(
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store.2)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store.3)),
    );
    let lo_s = _mm_packs_epi32(
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store.0)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store.1)),
    );
    _mm_packus_epi16(lo_s, hi_s)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_pack_ps_x2_epi8(store: (__m128, __m128)) -> __m128i {
    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    let lo_s = _mm_packs_epi32(
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store.0)),
        _mm_cvtps_epi32(_mm_round_ps::<ROUNDING_FLAGS>(store.1)),
    );
    _mm_packus_epi16(lo_s, _mm_setzero_si128())
}
