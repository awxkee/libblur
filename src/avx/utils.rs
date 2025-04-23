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
pub(crate) unsafe fn _mm256_mul_by_3_epi32(v: __m256i) -> __m256i {
    _mm256_add_epi32(_mm256_slli_epi32::<1>(v), v)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_opt_fnmlsf_ps<const FMA: bool>(
    a: __m256,
    b: __m256,
    c: __m256,
) -> __m256 {
    if FMA {
        _mm256_fmsub_ps(b, c, a)
    } else {
        _mm256_sub_ps(_mm256_mul_ps(b, c), a)
    }
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
        _mm256_add_ps(a, _mm256_mul_ps(b, c))
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm256_opt_fnmlaf_ps<const FMA: bool>(
    a: __m256,
    b: __m256,
    c: __m256,
) -> __m256 {
    if FMA {
        _mm256_fnmadd_ps(b, c, a)
    } else {
        _mm256_sub_ps(a, _mm256_mul_ps(b, c))
    }
}
