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
use crate::avx::{_mm256_pack_i32, _mm256_pack_u16};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_mul_epi8_by_ps_x4(
    input: __m256i,
    weight: __m256,
) -> (__m256, __m256, __m256, __m256) {
    let lo_16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(input));
    let hi_16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(input));

    (
        _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo_16))),
            weight,
        ),
        _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(lo_16))),
            weight,
        ),
        _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi_16))),
            weight,
        ),
        _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(hi_16))),
            weight,
        ),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_mul_epi8_by_epi16_x4(
    input: __m256i,
    weight: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let lo_16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(input));
    let hi_16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(input));

    (
        _mm256_madd_epi16(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo_16)), weight),
        _mm256_madd_epi16(
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(lo_16)),
            weight,
        ),
        _mm256_madd_epi16(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi_16)), weight),
        _mm256_madd_epi16(
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(hi_16)),
            weight,
        ),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_mul_add_epi8_by_epi16_x4(
    accumulator: (__m256i, __m256i, __m256i, __m256i),
    input: __m256i,
    weight: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let lo_16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(input));
    let hi_16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(input));

    (
        _mm256_add_epi32(
            accumulator.0,
            _mm256_madd_epi16(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo_16)), weight),
        ),
        _mm256_add_epi32(
            accumulator.1,
            _mm256_madd_epi16(
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(lo_16)),
                weight,
            ),
        ),
        _mm256_add_epi32(
            accumulator.2,
            _mm256_madd_epi16(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi_16)), weight),
        ),
        _mm256_add_epi32(
            accumulator.3,
            _mm256_madd_epi16(
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(hi_16)),
                weight,
            ),
        ),
    )
}

#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn _mm256_fmlaf_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_fmadd_ps(b, c, a)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_opt_fmlaf_ps<const FMA: bool>(a: __m256, b: __m256, c: __m256) -> __m256 {
    if FMA {
        _mm256_fmlaf_ps(a, b, c)
    } else {
        _mm256_add_ps(_mm256_mul_ps(b, c), a)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_mul_add_epi8_by_ps_x4<const FMA: bool>(
    accumulator: (__m256, __m256, __m256, __m256),
    input: __m256i,
    weight: __m256,
) -> (__m256, __m256, __m256, __m256) {
    let lo_16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(input));
    let hi_16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(input));

    (
        _mm256_opt_fmlaf_ps::<FMA>(
            accumulator.0,
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo_16))),
            weight,
        ),
        _mm256_opt_fmlaf_ps::<FMA>(
            accumulator.1,
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(lo_16))),
            weight,
        ),
        _mm256_opt_fmlaf_ps::<FMA>(
            accumulator.2,
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi_16))),
            weight,
        ),
        _mm256_opt_fmlaf_ps::<FMA>(
            accumulator.3,
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(hi_16))),
            weight,
        ),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_mul_add_symm_epi8_by_ps_x4<const FMA: bool>(
    accumulator: (__m256, __m256, __m256, __m256),
    input0: __m256i,
    input1: __m256i,
    weight: __m256,
) -> (__m256, __m256, __m256, __m256) {
    let lo_16 = _mm256_add_epi16(
        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(input0)),
        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(input1)),
    );
    let hi_16 = _mm256_add_epi16(
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(input0)),
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(input1)),
    );

    (
        _mm256_opt_fmlaf_ps::<FMA>(
            accumulator.0,
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo_16))),
            weight,
        ),
        _mm256_opt_fmlaf_ps::<FMA>(
            accumulator.1,
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(lo_16))),
            weight,
        ),
        _mm256_opt_fmlaf_ps::<FMA>(
            accumulator.2,
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi_16))),
            weight,
        ),
        _mm256_opt_fmlaf_ps::<FMA>(
            accumulator.3,
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(hi_16))),
            weight,
        ),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_pack_ps_x4_epi8(store: (__m256, __m256, __m256, __m256)) -> __m256i {
    const ROUNDING_FLAGS: i32 = 0;
    let hi_s = _mm256_pack_i32(
        _mm256_cvtps_epi32(_mm256_round_ps::<ROUNDING_FLAGS>(store.2)),
        _mm256_cvtps_epi32(_mm256_round_ps::<ROUNDING_FLAGS>(store.3)),
    );
    let lo_s = _mm256_pack_i32(
        _mm256_cvtps_epi32(_mm256_round_ps::<ROUNDING_FLAGS>(store.0)),
        _mm256_cvtps_epi32(_mm256_round_ps::<ROUNDING_FLAGS>(store.1)),
    );
    _mm256_pack_u16(lo_s, hi_s)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_pack_epi32_x4_epi8(store: (__m256i, __m256i, __m256i, __m256i)) -> __m256i {
    let rounding_const = _mm256_set1_epi32(1 << 14);
    let hi_s = _mm256_pack_i32(
        _mm256_srai_epi32::<15>(_mm256_add_epi32(store.2, rounding_const)),
        _mm256_srai_epi32::<15>(_mm256_add_epi32(store.3, rounding_const)),
    );
    let lo_s = _mm256_pack_i32(
        _mm256_srai_epi32::<15>(_mm256_add_epi32(store.0, rounding_const)),
        _mm256_srai_epi32::<15>(_mm256_add_epi32(store.1, rounding_const)),
    );
    _mm256_pack_u16(lo_s, hi_s)
}
