/*
 * // Copyright (c) Radzivon Bartoshyk 2/2025. All rights reserved.
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
pub(crate) unsafe fn _mm256_mul_round_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_fmadd_ps(a, b, _mm256_set1_ps(0.5f32))
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_round_ps(a: __m128, b: __m128) -> __m128 {
    _mm_fmadd_ps(a, b, _mm_set1_ps(0.5f32))
}

#[inline(always)]
pub(crate) unsafe fn _mm256_mulhi_epi32_m128(
    a0: __m128i,
    b0: __m256i,
    a1: __m128i,
    b1: __m256i,
) -> (__m128i, __m128i) {
    let rnd = _mm256_set1_epi64x((1 << 30) - 1);
    let perm0 = _mm256_setr_epi32(0, -1, 1, -1, 2, -1, 3, -1);
    let lo = _mm256_mul_epi32(
        _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(a0), perm0),
        b0,
    );
    let hi = _mm256_mul_epi32(
        _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(a1), perm0),
        b1,
    );

    let a0 = _mm256_add_epi64(lo, rnd);
    let a1 = _mm256_add_epi64(hi, rnd);

    let b0 = _mm256_srli_epi64::<31>(a0);
    let b1 = _mm256_srli_epi64::<31>(a1);

    let perm = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    let shuffled0 = _mm256_permutevar8x32_epi32(b0, perm);
    let shuffled1 = _mm256_permutevar8x32_epi32(b1, perm);
    (
        _mm256_castsi256_si128(shuffled0),
        _mm256_castsi256_si128(shuffled1),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_mulhi_epi32(a0: __m128i, b0: __m256i) -> __m128i {
    let perm0 = _mm256_setr_epi32(0, -1, 1, -1, 2, -1, 3, -1);

    let rnd = _mm256_set1_epi64x((1 << 30) - 1);
    let lo = _mm256_mul_epi32(
        _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(a0), perm0),
        b0,
    );

    let a0 = _mm256_add_epi64(lo, rnd);

    let b0 = _mm256_srli_epi64::<31>(a0);

    let perm = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    let shuffled0 = _mm256_permutevar8x32_epi32(b0, perm);
    _mm256_castsi256_si128(shuffled0)
}
