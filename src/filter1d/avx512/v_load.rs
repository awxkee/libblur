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
pub(crate) unsafe fn _mm256_load_pack_x4(ptr: *const u8) -> (__m256i, __m256i, __m256i, __m256i) {
    let row0 = _mm256_loadu_si256(ptr as *const __m256i);
    let row1 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
    let row2 = _mm256_loadu_si256(ptr.add(64) as *const __m256i);
    let row3 = _mm256_loadu_si256(ptr.add(96) as *const __m256i);
    (row0, row1, row2, row3)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_load_pack_x3(ptr: *const u8) -> (__m256i, __m256i, __m256i) {
    let row0 = _mm256_loadu_si256(ptr as *const __m256i);
    let row1 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
    let row2 = _mm256_loadu_si256(ptr.add(64) as *const __m256i);
    (row0, row1, row2)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_load_pack_x2(ptr: *const u8) -> (__m256i, __m256i) {
    let row0 = _mm256_loadu_si256(ptr as *const __m256i);
    let row1 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
    (row0, row1)
}

#[inline(always)]
pub(crate) unsafe fn _mm512_load_pack_x2(ptr: *const u8) -> (__m512i, __m512i) {
    let row0 = _mm512_loadu_si512(ptr as *const _);
    let row1 = _mm512_loadu_si512(ptr.add(64) as *const _);
    (row0, row1)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_load_deinterleave_rgb(ptr: *const u8) -> (__m256i, __m256i, __m256i) {
    let row0 = _mm256_loadu_si256(ptr as *const __m256i);
    let row1 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
    let row2 = _mm256_loadu_si256(ptr.add(64) as *const __m256i);
    _mm256_deinterleave_rgb(row0, row1, row2)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_load_deinterleave_rgba(
    ptr: *const u8,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let row0 = _mm256_loadu_si256(ptr as *const __m256i);
    let row1 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
    let row2 = _mm256_loadu_si256(ptr.add(64) as *const __m256i);
    let row3 = _mm256_loadu_si256(ptr.add(96) as *const __m256i);
    _mm256_deinterleave_rgba_epi8(row0, row1, row2, row3)
}

#[inline(always)]
pub(crate) unsafe fn _mm512_load_pack_ps_x4(ptr: *const f32) -> (__m512, __m512, __m512, __m512) {
    let row0 = _mm512_loadu_ps(ptr);
    let row1 = _mm512_loadu_ps(ptr.add(16));
    let row2 = _mm512_loadu_ps(ptr.add(32));
    let row3 = _mm512_loadu_ps(ptr.add(48));
    (row0, row1, row2, row3)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_load_pack_ps_x4(ptr: *const f32) -> (__m256, __m256, __m256, __m256) {
    let row0 = _mm256_loadu_ps(ptr);
    let row1 = _mm256_loadu_ps(ptr.add(8));
    let row2 = _mm256_loadu_ps(ptr.add(16));
    let row3 = _mm256_loadu_ps(ptr.add(24));
    (row0, row1, row2, row3)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_load_pack_ps_x2(ptr: *const f32) -> (__m256, __m256) {
    let row0 = _mm256_loadu_ps(ptr);
    let row1 = _mm256_loadu_ps(ptr.add(8));
    (row0, row1)
}

#[inline(always)]
pub(crate) unsafe fn _mm512_load_pack_ps_x2(ptr: *const f32) -> (__m512, __m512) {
    let row0 = _mm512_loadu_ps(ptr);
    let row1 = _mm512_loadu_ps(ptr.add(16));
    (row0, row1)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_load_deinterleave_rgb_ps(ptr: *const f32) -> (__m256, __m256, __m256) {
    let row0 = _mm256_loadu_ps(ptr);
    let row1 = _mm256_loadu_ps(ptr.add(8));
    let row2 = _mm256_loadu_ps(ptr.add(16));
    _mm256_deinterleave_rgb_ps((row0, row1, row2))
}

#[inline(always)]
pub(crate) unsafe fn _mm256_load_deinterleave_rgba_ps(
    ptr: *const f32,
) -> (__m256, __m256, __m256, __m256) {
    let row0 = _mm256_loadu_ps(ptr);
    let row1 = _mm256_loadu_ps(ptr.add(8));
    let row2 = _mm256_loadu_ps(ptr.add(16));
    let row3 = _mm256_loadu_ps(ptr.add(24));
    _mm256_deinterleave_rgba_ps((row0, row1, row2, row3))
}
