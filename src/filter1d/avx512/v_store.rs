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

use crate::filter1d::avx512::pack::*;
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn _mm256_store_pack_x4(ptr: *mut u8, v: (__m256i, __m256i, __m256i, __m256i)) {
    _mm256_storeu_si256(ptr as *mut __m256i, v.0);
    _mm256_storeu_si256(ptr.add(32) as *mut __m256i, v.1);
    _mm256_storeu_si256(ptr.add(64) as *mut __m256i, v.2);
    _mm256_storeu_si256(ptr.add(96) as *mut __m256i, v.3);
}

#[inline(always)]
pub(crate) unsafe fn _mm256_store_pack_x2(ptr: *mut u8, v: (__m256i, __m256i)) {
    _mm256_storeu_si256(ptr as *mut __m256i, v.0);
    _mm256_storeu_si256(ptr.add(32) as *mut __m256i, v.1);
}

#[inline(always)]
pub(crate) unsafe fn _mm512_store_pack_x2(ptr: *mut u8, v: (__m512i, __m512i)) {
    _mm512_storeu_si512(ptr as *mut __m512i, v.0);
    _mm512_storeu_si512(ptr.add(64) as *mut __m512i, v.1);
}

#[inline(always)]
pub(crate) unsafe fn _mm256_store_pack_x3(ptr: *mut u8, v: (__m256i, __m256i, __m256i)) {
    _mm256_storeu_si256(ptr as *mut __m256i, v.0);
    _mm256_storeu_si256(ptr.add(32) as *mut __m256i, v.1);
    _mm256_storeu_si256(ptr.add(64) as *mut __m256i, v.2);
}

#[inline(always)]
pub(crate) unsafe fn _mm256_store_pack_x3_ps(ptr: *mut f32, v: (__m256, __m256, __m256)) {
    _mm256_storeu_ps(ptr, v.0);
    _mm256_storeu_ps(ptr.add(8), v.1);
    _mm256_storeu_ps(ptr.add(16), v.2);
}

#[inline(always)]
pub(crate) unsafe fn _mm256_store_interleave_rgb(ptr: *mut u8, v: (__m256i, __m256i, __m256i)) {
    let (v0, v1, v2) = _mm256_interleave_rgb(v.0, v.1, v.2);
    _mm256_store_pack_x3(ptr, (v0, v1, v2));
}

#[inline(always)]
pub(crate) unsafe fn _mm256_store_interleave_rgba(
    ptr: *mut u8,
    v: (__m256i, __m256i, __m256i, __m256i),
) {
    let (v0, v1, v2, v3) = _mm256_interleave_rgba_epi8((v.0, v.1, v.2, v.3));
    _mm256_store_pack_x4(ptr, (v0, v1, v2, v3));
}

#[inline(always)]
pub(crate) unsafe fn _mm256_store_pack_ps_x4(ptr: *mut f32, v: (__m256, __m256, __m256, __m256)) {
    _mm256_storeu_ps(ptr, v.0);
    _mm256_storeu_ps(ptr.add(8), v.1);
    _mm256_storeu_ps(ptr.add(16), v.2);
    _mm256_storeu_ps(ptr.add(24), v.3);
}

#[inline(always)]
pub(crate) unsafe fn _mm512_store_pack_ps_x4(ptr: *mut f32, v: (__m512, __m512, __m512, __m512)) {
    _mm512_storeu_ps(ptr, v.0);
    _mm512_storeu_ps(ptr.add(16), v.1);
    _mm512_storeu_ps(ptr.add(32), v.2);
    _mm512_storeu_ps(ptr.add(48), v.3);
}

#[inline(always)]
pub(crate) unsafe fn _mm256_store_pack_ps_x2(ptr: *mut f32, v: (__m256, __m256)) {
    _mm256_storeu_ps(ptr, v.0);
    _mm256_storeu_ps(ptr.add(8), v.1);
}

#[inline(always)]
pub(crate) unsafe fn _mm512_store_pack_ps_x2(ptr: *mut f32, v: (__m512, __m512)) {
    _mm512_storeu_ps(ptr, v.0);
    _mm512_storeu_ps(ptr.add(16), v.1);
}

#[inline(always)]
pub(crate) unsafe fn _mm256_store_interleave_rgb_ps(ptr: *mut f32, v: (__m256, __m256, __m256)) {
    let (v0, v1, v2) = _mm256_interleave_rgb_ps(v);
    _mm256_store_pack_x3_ps(ptr, (v0, v1, v2));
}

#[inline(always)]
pub(crate) unsafe fn _mm256_store_interleave_rgba_ps(
    ptr: *mut f32,
    v: (__m256, __m256, __m256, __m256),
) {
    let (v0, v1, v2, v3) = _mm256_interleave_rgba_ps(v);
    _mm256_store_pack_ps_x4(ptr, (v0, v1, v2, v3));
}
