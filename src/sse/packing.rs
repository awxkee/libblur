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

use crate::sse::_shuffle;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn _mm_deinterleave_rgba(
    rgba0: __m128i,
    rgba1: __m128i,
    rgba2: __m128i,
    rgba3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let t0 = _mm_unpacklo_epi8(rgba0, rgba1); // r1 R1 g1 G1 b1 B1 a1 A1 r2 R2 g2 G2 b2 B2 a2 A2
    let t1 = _mm_unpackhi_epi8(rgba0, rgba1);
    let t2 = _mm_unpacklo_epi8(rgba2, rgba3); // r4 R4 g4 G4 b4 B4 a4 A4 r5 R5 g5 G5 b5 B5 a5 A5
    let t3 = _mm_unpackhi_epi8(rgba2, rgba3);

    let t4 = _mm_unpacklo_epi16(t0, t2); // r1 R1 r4 R4 g1 G1 G4 g4 G4 b1 B1 b4 B4 a1 A1 a4 A4
    let t5 = _mm_unpackhi_epi16(t0, t2);
    let t6 = _mm_unpacklo_epi16(t1, t3);
    let t7 = _mm_unpackhi_epi16(t1, t3);

    let l1 = _mm_unpacklo_epi32(t4, t6); // r1 R1 r4 R4 g1 G1 G4 g4 G4 b1 B1 b4 B4 a1 A1 a4 A4
    let l2 = _mm_unpackhi_epi32(t4, t6);
    let l3 = _mm_unpacklo_epi32(t5, t7);
    let l4 = _mm_unpackhi_epi32(t5, t7);

    #[rustfmt::skip]
    let shuffle = _mm_setr_epi8(0, 4, 8, 12,
                                        1, 5, 9, 13,
                                        2, 6, 10, 14,
                                        3, 7, 11, 15,
    );

    let r1 = _mm_shuffle_epi8(_mm_unpacklo_epi32(l1, l3), shuffle);
    let r2 = _mm_shuffle_epi8(_mm_unpackhi_epi32(l1, l3), shuffle);
    let r3 = _mm_shuffle_epi8(_mm_unpacklo_epi32(l2, l4), shuffle);
    let r4 = _mm_shuffle_epi8(_mm_unpackhi_epi32(l2, l4), shuffle);

    (r1, r2, r3, r4)
}

#[inline(always)]
pub(crate) unsafe fn _mm_deinterleave_rgb(
    s0: __m128i,
    s1: __m128i,
    s2: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
    let m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);

    let a0 = _mm_blendv_epi8(_mm_blendv_epi8(s0, s1, m0), s2, m1);
    let b0 = _mm_blendv_epi8(_mm_blendv_epi8(s1, s2, m0), s0, m1);
    let c0 = _mm_blendv_epi8(_mm_blendv_epi8(s2, s0, m0), s1, m1);
    let sh_b = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
    let sh_g = _mm_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14);
    let sh_r = _mm_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);
    let a0 = _mm_shuffle_epi8(a0, sh_b);
    let b0 = _mm_shuffle_epi8(b0, sh_g);
    let c0 = _mm_shuffle_epi8(c0, sh_r);

    (a0, b0, c0)
}

#[inline(always)]
pub(crate) unsafe fn _mm_interleave_rgb(
    r: __m128i,
    g: __m128i,
    b: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let sh_a = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
    let sh_b = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
    let sh_c = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
    let a0 = _mm_shuffle_epi8(r, sh_a);
    let b0 = _mm_shuffle_epi8(g, sh_b);
    let c0 = _mm_shuffle_epi8(b, sh_c);

    let m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
    let m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    let v0 = _mm_blendv_epi8(_mm_blendv_epi8(a0, b0, m1), c0, m0);
    let v1 = _mm_blendv_epi8(_mm_blendv_epi8(b0, c0, m1), a0, m0);
    let v2 = _mm_blendv_epi8(_mm_blendv_epi8(c0, a0, m1), b0, m0);
    (v0, v1, v2)
}

#[inline(always)]
pub(crate) unsafe fn _mm_interleave_rgba(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    a: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let rg_lo = _mm_unpacklo_epi8(r, g);
    let rg_hi = _mm_unpackhi_epi8(r, g);
    let ba_lo = _mm_unpacklo_epi8(b, a);
    let ba_hi = _mm_unpackhi_epi8(b, a);

    let rgba_0_lo = _mm_unpacklo_epi16(rg_lo, ba_lo);
    let rgba_0_hi = _mm_unpackhi_epi16(rg_lo, ba_lo);
    let rgba_1_lo = _mm_unpacklo_epi16(rg_hi, ba_hi);
    let rgba_1_hi = _mm_unpackhi_epi16(rg_hi, ba_hi);
    (rgba_0_lo, rgba_0_hi, rgba_1_lo, rgba_1_hi)
}

#[inline(always)]
pub(crate) unsafe fn _mm_deinterleave_rgba_ps(
    t0: __m128,
    t1: __m128,
    t2: __m128,
    t3: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let t02lo = _mm_unpacklo_ps(t0, t2);
    let t13lo = _mm_unpacklo_ps(t1, t3);
    let t02hi = _mm_unpackhi_ps(t0, t2);
    let t13hi = _mm_unpackhi_ps(t1, t3);
    let v0 = _mm_unpacklo_ps(t02lo, t13lo);
    let v1 = _mm_unpackhi_ps(t02lo, t13lo);
    let v2 = _mm_unpacklo_ps(t02hi, t13hi);
    let v3 = _mm_unpackhi_ps(t02hi, t13hi);
    (v0, v1, v2, v3)
}

#[inline(always)]
pub(crate) unsafe fn _mm_deinterleave_rgb_ps(
    t0: __m128,
    t1: __m128,
    t2: __m128,
) -> (__m128, __m128, __m128) {
    const FLAG_1: i32 = _shuffle(0, 1, 0, 2);
    let at12 = _mm_shuffle_ps::<FLAG_1>(t1, t2);
    const FLAG_2: i32 = _shuffle(2, 0, 3, 0);
    let v0 = _mm_shuffle_ps::<FLAG_2>(t0, at12);
    const FLAG_3: i32 = _shuffle(0, 0, 0, 1);
    let bt01 = _mm_shuffle_ps::<FLAG_3>(t0, t1);
    const FLAG_4: i32 = _shuffle(0, 2, 0, 3);
    let bt12 = _mm_shuffle_ps::<FLAG_4>(t1, t2);
    const FLAG_5: i32 = _shuffle(2, 0, 2, 0);
    let v1 = _mm_shuffle_ps::<FLAG_5>(bt01, bt12);

    const FLAG_6: i32 = _shuffle(0, 1, 0, 2);
    let ct01 = _mm_shuffle_ps::<FLAG_6>(t0, t1);
    const FLAG_7: i32 = _shuffle(3, 0, 2, 0);
    let v2 = _mm_shuffle_ps::<FLAG_7>(ct01, t2);
    (v0, v1, v2)
}

#[inline(always)]
pub(crate) unsafe fn _mm_interleave_rgb_ps(
    t0: __m128,
    t1: __m128,
    t2: __m128,
) -> (__m128, __m128, __m128) {
    const FLAG_1: i32 = _shuffle(0, 0, 0, 0);
    let u0 = _mm_shuffle_ps::<FLAG_1>(t0, t1);
    const FLAG_2: i32 = _shuffle(1, 1, 0, 0);
    let u1 = _mm_shuffle_ps::<FLAG_2>(t2, t0);
    const FLAG_3: i32 = _shuffle(2, 0, 2, 0);
    let v0 = _mm_shuffle_ps::<FLAG_3>(u0, u1);
    const FLAG_4: i32 = _shuffle(1, 1, 1, 1);
    let u2 = _mm_shuffle_ps::<FLAG_4>(t1, t2);
    const FLAG_5: i32 = _shuffle(2, 2, 2, 2);
    let u3 = _mm_shuffle_ps::<FLAG_5>(t0, t1);
    const FLAG_6: i32 = _shuffle(2, 0, 2, 0);
    let v1 = _mm_shuffle_ps::<FLAG_6>(u2, u3);
    const FLAG_7: i32 = _shuffle(3, 3, 2, 2);
    let u4 = _mm_shuffle_ps::<FLAG_7>(t2, t0);
    const FLAG_8: i32 = _shuffle(3, 3, 3, 3);
    let u5 = _mm_shuffle_ps::<FLAG_8>(t1, t2);
    const FLAG_9: i32 = _shuffle(2, 0, 2, 0);
    let v2 = _mm_shuffle_ps::<FLAG_9>(u4, u5);
    (v0, v1, v2)
}

#[inline(always)]
pub(crate) unsafe fn _mm_interleave_rgba_ps(
    t0: __m128,
    t1: __m128,
    t2: __m128,
    t3: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let u0 = _mm_unpacklo_ps(t0, t2);
    let u1 = _mm_unpacklo_ps(t1, t3);
    let u2 = _mm_unpackhi_ps(t0, t2);
    let u3 = _mm_unpackhi_ps(t1, t3);
    let v0 = _mm_unpacklo_ps(u0, u1);
    let v2 = _mm_unpacklo_ps(u2, u3);
    let v1 = _mm_unpackhi_ps(u0, u1);
    let v3 = _mm_unpackhi_ps(u2, u3);
    (v0, v1, v2, v3)
}
