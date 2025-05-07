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
pub(crate) const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
pub(crate) unsafe fn _mm256_pack_u16(s_1: __m256i, s_2: __m256i) -> __m256i {
    let packed = _mm256_packus_epi16(s_1, s_2);
    const MASK: i32 = shuffle(3, 1, 2, 0);
    _mm256_permute4x64_epi64::<MASK>(packed)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_deinterleave_rgba_epi8(
    rgba0: __m256i,
    rgba1: __m256i,
    rgba2: __m256i,
    rgba3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    #[rustfmt::skip]
    let sh = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5,
        9, 13, 2, 6, 10, 14,
        3, 7, 11, 15, 0, 4,
        8, 12, 1, 5, 9, 13,
        2, 6, 10, 14, 3, 7,
        11, 15,
    );

    let p0 = _mm256_shuffle_epi8(rgba0, sh);
    let p1 = _mm256_shuffle_epi8(rgba1, sh);
    let p2 = _mm256_shuffle_epi8(rgba2, sh);
    let p3 = _mm256_shuffle_epi8(rgba3, sh);

    let p01l = _mm256_unpacklo_epi32(p0, p1);
    let p01h = _mm256_unpackhi_epi32(p0, p1);
    let p23l = _mm256_unpacklo_epi32(p2, p3);
    let p23h = _mm256_unpackhi_epi32(p2, p3);

    let pll = _mm256_permute2x128_si256::<32>(p01l, p23l);
    let plh = _mm256_permute2x128_si256::<49>(p01l, p23l);
    let phl = _mm256_permute2x128_si256::<32>(p01h, p23h);
    let phh = _mm256_permute2x128_si256::<49>(p01h, p23h);

    let b0 = _mm256_unpacklo_epi32(pll, plh);
    let g0 = _mm256_unpackhi_epi32(pll, plh);
    let r0 = _mm256_unpacklo_epi32(phl, phh);
    let a0 = _mm256_unpackhi_epi32(phl, phh);

    (b0, g0, r0, a0)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_interleave_rgba_epi8(
    vals: (__m256i, __m256i, __m256i, __m256i),
) -> (__m256i, __m256i, __m256i, __m256i) {
    let bg0 = _mm256_unpacklo_epi8(vals.0, vals.1);
    let bg1 = _mm256_unpackhi_epi8(vals.0, vals.1);
    let ra0 = _mm256_unpacklo_epi8(vals.2, vals.3);
    let ra1 = _mm256_unpackhi_epi8(vals.2, vals.3);

    let rgba0_ = _mm256_unpacklo_epi16(bg0, ra0);
    let rgba1_ = _mm256_unpackhi_epi16(bg0, ra0);
    let rgba2_ = _mm256_unpacklo_epi16(bg1, ra1);
    let rgba3_ = _mm256_unpackhi_epi16(bg1, ra1);

    let rgba0 = _mm256_permute2x128_si256::<32>(rgba0_, rgba1_);
    let rgba2 = _mm256_permute2x128_si256::<49>(rgba0_, rgba1_);
    let rgba1 = _mm256_permute2x128_si256::<32>(rgba2_, rgba3_);
    let rgba3 = _mm256_permute2x128_si256::<49>(rgba2_, rgba3_);
    (rgba0, rgba1, rgba2, rgba3)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_interleave_rgb(
    r: __m256i,
    g: __m256i,
    b: __m256i,
) -> (__m256i, __m256i, __m256i) {
    let sh_b = _mm256_setr_epi8(
        0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14,
        9, 4, 15, 10, 5,
    );
    let sh_g = _mm256_setr_epi8(
        5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3,
        14, 9, 4, 15, 10,
    );
    let sh_r = _mm256_setr_epi8(
        10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8,
        3, 14, 9, 4, 15,
    );

    let b0 = _mm256_shuffle_epi8(r, sh_b);
    let g0 = _mm256_shuffle_epi8(g, sh_g);
    let r0 = _mm256_shuffle_epi8(b, sh_r);

    let m0 = _mm256_setr_epi8(
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
    );
    let m1 = _mm256_setr_epi8(
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
    );

    let p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, g0, m0), r0, m1);
    let p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, r0, m0), b0, m1);
    let p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, b0, m0), g0, m1);

    let bgr0 = _mm256_permute2x128_si256::<32>(p0, p1);
    let bgr1 = _mm256_permute2x128_si256::<48>(p2, p0);
    let bgr2 = _mm256_permute2x128_si256::<49>(p1, p2);

    (bgr0, bgr1, bgr2)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_deinterleave_rgb(
    rgb0: __m256i,
    rgb1: __m256i,
    rgb2: __m256i,
) -> (__m256i, __m256i, __m256i) {
    let s02_low = _mm256_permute2x128_si256::<32>(rgb0, rgb2);
    let s02_high = _mm256_permute2x128_si256::<49>(rgb0, rgb2);

    #[rustfmt::skip]
    let m0 = _mm256_setr_epi8(
        0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
        0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
        0, -1, 0, 0, -1,
        0, 0,
    );

    #[rustfmt::skip]
    let m1 = _mm256_setr_epi8(
        0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
        0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
        0, -1,
    );

    let b0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, m0), rgb1, m1);
    let g0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, m1), rgb1, m0);
    let r0 = _mm256_blendv_epi8(_mm256_blendv_epi8(rgb1, s02_low, m0), s02_high, m1);

    #[rustfmt::skip]
    let sh_b = _mm256_setr_epi8(
        0, 3, 6, 9, 12,
        15, 2, 5, 8, 11,
        14, 1, 4, 7, 10,
        13, 0, 3, 6, 9,
        12, 15, 2, 5, 8,
        11, 14, 1, 4, 7,
        10, 13,
    );

    #[rustfmt::skip]
    let sh_g = _mm256_setr_epi8(
        1, 4, 7, 10, 13,
        0, 3, 6, 9, 12,
        15, 2, 5, 8, 11,
        14, 1, 4, 7, 10,
        13, 0, 3, 6, 9,
        12, 15, 2, 5, 8,
        11, 14,
    );

    #[rustfmt::skip]
    let sh_r = _mm256_setr_epi8(
        2, 5, 8, 11, 14,
        1, 4, 7, 10, 13,
        0, 3, 6, 9, 12,
        15, 2, 5, 8, 11,
        14, 1, 4, 7, 10,
        13, 0, 3, 6, 9,
        12, 15,
    );
    let b0 = _mm256_shuffle_epi8(b0, sh_b);
    let g0 = _mm256_shuffle_epi8(g0, sh_g);
    let r0 = _mm256_shuffle_epi8(r0, sh_r);
    (b0, g0, r0)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_deinterleave_rgba_epi32(
    vals: (__m256i, __m256i, __m256i, __m256i),
) -> (__m256i, __m256i, __m256i, __m256i) {
    let p01l = _mm256_unpacklo_epi32(vals.0, vals.1);
    let p01h = _mm256_unpackhi_epi32(vals.0, vals.1);
    let p23l = _mm256_unpacklo_epi32(vals.2, vals.3);
    let p23h = _mm256_unpackhi_epi32(vals.2, vals.3);

    let pll = _mm256_permute2x128_si256::<32>(p01l, p23l);
    let plh = _mm256_permute2x128_si256::<49>(p01l, p23l);
    let phl = _mm256_permute2x128_si256::<32>(p01h, p23h);
    let phh = _mm256_permute2x128_si256::<49>(p01h, p23h);

    let v0 = _mm256_unpacklo_epi32(pll, plh);
    let v1 = _mm256_unpackhi_epi32(pll, plh);
    let v2 = _mm256_unpacklo_epi32(phl, phh);
    let v3 = _mm256_unpackhi_epi32(phl, phh);
    (v0, v1, v2, v3)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_deinterleave_rgb_epi32(
    vals: (__m256i, __m256i, __m256i),
) -> (__m256i, __m256i, __m256i) {
    let s02_low = _mm256_permute2x128_si256::<32>(vals.0, vals.2);
    let s02_high = _mm256_permute2x128_si256::<49>(vals.0, vals.2);

    let b0 = _mm256_blend_epi32::<0x92>(_mm256_blend_epi32::<0x24>(s02_low, s02_high), vals.1);
    let g0 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(s02_high, s02_low), vals.1);
    let r0 = _mm256_blend_epi32::<0x92>(_mm256_blend_epi32::<0x24>(vals.1, s02_low), s02_high);

    let v0 = _mm256_shuffle_epi32::<0x6c>(b0);
    let v1 = _mm256_shuffle_epi32::<0xb1>(g0);
    let v2 = _mm256_shuffle_epi32::<0xc6>(r0);
    (v0, v1, v2)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_deinterleave_rgba_ps(
    vals: (__m256, __m256, __m256, __m256),
) -> (__m256, __m256, __m256, __m256) {
    let (v0, v1, v2, v3) = _mm256_deinterleave_rgba_epi32((
        _mm256_castps_si256(vals.0),
        _mm256_castps_si256(vals.1),
        _mm256_castps_si256(vals.2),
        _mm256_castps_si256(vals.3),
    ));
    (
        _mm256_castsi256_ps(v0),
        _mm256_castsi256_ps(v1),
        _mm256_castsi256_ps(v2),
        _mm256_castsi256_ps(v3),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm256_deinterleave_rgb_ps(
    vals: (__m256, __m256, __m256),
) -> (__m256, __m256, __m256) {
    let (v0, v1, v2) = _mm256_deinterleave_rgb_epi32((
        _mm256_castps_si256(vals.0),
        _mm256_castps_si256(vals.1),
        _mm256_castps_si256(vals.2),
    ));
    (
        _mm256_castsi256_ps(v0),
        _mm256_castsi256_ps(v1),
        _mm256_castsi256_ps(v2),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm256_interleave_rgb_epi32(
    vals: (__m256i, __m256i, __m256i),
) -> (__m256i, __m256i, __m256i) {
    let b0 = _mm256_shuffle_epi32::<0x6c>(vals.0);
    let g0 = _mm256_shuffle_epi32::<0xb1>(vals.1);
    let r0 = _mm256_shuffle_epi32::<0xc6>(vals.2);

    let p0 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(b0, g0), r0);
    let p1 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(g0, r0), b0);
    let p2 = _mm256_blend_epi32::<0x24>(_mm256_blend_epi32::<0x92>(r0, b0), g0);

    let v0 = _mm256_permute2x128_si256::<32>(p0, p1);
    let v1 = p2;
    let v2 = _mm256_permute2x128_si256::<49>(p0, p1);
    (v0, v1, v2)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_interleave_rgb_ps(
    vals: (__m256, __m256, __m256),
) -> (__m256, __m256, __m256) {
    let (v0, v1, v2) = _mm256_interleave_rgb_epi32((
        _mm256_castps_si256(vals.0),
        _mm256_castps_si256(vals.1),
        _mm256_castps_si256(vals.2),
    ));
    (
        _mm256_castsi256_ps(v0),
        _mm256_castsi256_ps(v1),
        _mm256_castsi256_ps(v2),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm256_interleave_rgba_epi32(
    vals: (__m256i, __m256i, __m256i, __m256i),
) -> (__m256i, __m256i, __m256i, __m256i) {
    let bg0 = _mm256_unpacklo_epi32(vals.0, vals.1);
    let bg1 = _mm256_unpackhi_epi32(vals.0, vals.1);
    let ra0 = _mm256_unpacklo_epi32(vals.2, vals.3);
    let ra1 = _mm256_unpackhi_epi32(vals.2, vals.3);

    let bgra0_ = _mm256_unpacklo_epi64(bg0, ra0);
    let bgra1_ = _mm256_unpackhi_epi64(bg0, ra0);
    let bgra2_ = _mm256_unpacklo_epi64(bg1, ra1);
    let bgra3_ = _mm256_unpackhi_epi64(bg1, ra1);

    let v0 = _mm256_permute2x128_si256::<32>(bgra0_, bgra1_);
    let v1 = _mm256_permute2x128_si256::<32>(bgra2_, bgra3_);
    let v2 = _mm256_permute2x128_si256::<49>(bgra0_, bgra1_);
    let v3 = _mm256_permute2x128_si256::<49>(bgra2_, bgra3_);

    (v0, v1, v2, v3)
}

#[inline(always)]
pub(crate) unsafe fn _mm256_interleave_rgba_ps(
    vals: (__m256, __m256, __m256, __m256),
) -> (__m256, __m256, __m256, __m256) {
    let (v0, v1, v2, v3) = _mm256_interleave_rgba_epi32((
        _mm256_castps_si256(vals.0),
        _mm256_castps_si256(vals.1),
        _mm256_castps_si256(vals.2),
        _mm256_castps_si256(vals.3),
    ));
    (
        _mm256_castsi256_ps(v0),
        _mm256_castsi256_ps(v1),
        _mm256_castsi256_ps(v2),
        _mm256_castsi256_ps(v3),
    )
}
