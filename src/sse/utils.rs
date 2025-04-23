// Copyright (c) Radzivon Bartoshyk. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#![allow(dead_code)]
use half::f16;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn load_f32<const CN: usize>(ptr: *const f32) -> __m128 {
    if CN == 4 {
        return _mm_loadu_ps(ptr);
    } else if CN == 3 {
        let mut j = _mm_loadu_si64(ptr as *const _);
        j = _mm_insert_epi32::<2>(j, ptr.add(2).read_unaligned().to_bits() as i32);
        return _mm_castsi128_ps(j);
    } else if CN == 2 {
        return _mm_castsi128_ps(_mm_loadu_si64(ptr as *const _));
    }
    _mm_castsi128_ps(_mm_loadu_si32(ptr as *const _))
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn load_u8_s16_fast<const CN: usize>(ptr: *const u8) -> __m128i {
    let sh1 = _mm_setr_epi8(0, -1, 1, -1, 2, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    if CN == 4 {
        let v = _mm_loadu_si32(ptr as *const _);
        _mm_shuffle_epi8(v, sh1)
    } else if CN == 3 {
        let mut v0 = _mm_loadu_si16(ptr);
        v0 = _mm_insert_epi8::<2>(v0, ptr.add(2).read_unaligned() as i32);
        _mm_shuffle_epi8(v0, sh1)
    } else if CN == 2 {
        let v0 = _mm_loadu_si16(ptr);
        _mm_shuffle_epi8(v0, sh1)
    } else {
        _mm_setr_epi16(ptr.read_unaligned() as i16, 0, 0, 0, 0, 0, 0, 0)
    }
}

#[inline(always)]
pub(crate) unsafe fn load_u16_s32_fast<const CN: usize>(ptr: *const u16) -> __m128i {
    let sh1 = _mm_setr_epi8(0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1, 6, 7, -1, -1);
    if CN == 4 {
        let v = _mm_loadu_si64(ptr as *const _);
        _mm_shuffle_epi8(v, sh1)
    } else if CN == 3 {
        let mut v0 = _mm_loadu_si32(ptr as *const _);
        v0 = _mm_insert_epi16::<2>(v0, ptr.add(2).read_unaligned() as i32);
        _mm_shuffle_epi8(v0, sh1)
    } else if CN == 2 {
        let v0 = _mm_loadu_si32(ptr as *const _);
        _mm_shuffle_epi8(v0, sh1)
    } else {
        _mm_shuffle_epi8(_mm_loadu_si16(ptr as *const _), sh1)
    }
}

#[inline(always)]
pub(crate) unsafe fn load_u8_s32_fast<const CN: usize>(ptr: *const u8) -> __m128i {
    let sh1 = _mm_setr_epi8(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1);
    if CN == 4 {
        let v = _mm_loadu_si32(ptr as *const _);
        _mm_shuffle_epi8(v, sh1)
    } else if CN == 3 {
        let mut v0 = _mm_loadu_si16(ptr);
        v0 = _mm_insert_epi8::<2>(v0, ptr.add(2).read_unaligned() as i32);
        _mm_shuffle_epi8(v0, sh1)
    } else if CN == 2 {
        let v0 = _mm_loadu_si16(ptr);
        _mm_shuffle_epi8(v0, sh1)
    } else {
        _mm_setr_epi32(ptr.read_unaligned() as i32, 0, 0, 0)
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_ps_epi32(ab: __m128i, cd: __m128) -> __m128i {
    let cvt = _mm_cvtepi32_ps(ab);
    let rs = _mm_mul_ps(cvt, cd);
    _mm_cvtps_epi32(rs)
}

#[inline(always)]
pub(crate) unsafe fn _mm_blendv_epi32x(xmm0: __m128i, xmm1: __m128i, mask: __m128i) -> __m128i {
    _mm_castps_si128(_mm_blendv_ps(
        _mm_castsi128_ps(xmm0),
        _mm_castsi128_ps(xmm1),
        _mm_castsi128_ps(mask),
    ))
}

pub(crate) const fn _shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
pub(crate) unsafe fn _mm_packus_epi64(a: __m128i, b: __m128i) -> __m128i {
    const SHUFFLE_MASK: i32 = _shuffle(3, 1, 2, 0);
    let a = _mm_shuffle_epi32::<SHUFFLE_MASK>(a);
    let b1 = _mm_shuffle_epi32::<SHUFFLE_MASK>(b);
    _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b1)))
}

#[inline(always)]
pub(crate) unsafe fn store_f32<const CN: usize>(dst_ptr: *mut f32, regi: __m128) {
    if CN == 4 {
        _mm_storeu_ps(dst_ptr, regi);
    } else if CN == 3 {
        _mm_storeu_si64(dst_ptr as *mut u8, _mm_castps_si128(regi));
        dst_ptr
            .add(2)
            .write_unaligned(f32::from_bits(_mm_extract_ps::<2>(regi) as u32));
    } else if CN == 2 {
        _mm_storeu_si64(dst_ptr as *mut u8, _mm_castps_si128(regi));
    } else {
        _mm_store_ss(dst_ptr as *mut _, regi);
    }
}

/// Stores u32 up to x4 as u8 up to x4 based on channels count
#[inline(always)]
#[cfg(any(feature = "sse", feature = "avx"))]
pub(crate) unsafe fn store_u8_s32<const CN: usize>(dst_ptr: *mut u8, regi: __m128i) {
    let s16 = _mm_packs_epi32(regi, regi);
    let v8 = _mm_packus_epi16(s16, s16);
    if CN == 4 {
        _mm_storeu_si32(dst_ptr as *mut _, v8);
    } else if CN == 3 {
        let pixel_3 = _mm_extract_epi8::<2>(v8);
        _mm_storeu_si16(dst_ptr, v8);
        dst_ptr.add(2).write_unaligned(pixel_3 as u8);
    } else if CN == 2 {
        _mm_storeu_si16(dst_ptr, v8);
    } else {
        let pixel_s32 = _mm_extract_epi32::<0>(v8);
        let pixel_bytes = pixel_s32.to_le_bytes();
        dst_ptr.write_unaligned(pixel_bytes[0]);
    }
}

/// Stores u32 up to x4 as u8 up to x4 based on channels count
#[inline(always)]
pub(crate) unsafe fn store_u8_u32<const CN: usize>(dst_ptr: *mut u8, regi: __m128i) {
    let s16 = _mm_packus_epi32(regi, regi);
    let v8 = _mm_packus_epi16(s16, s16);
    if CN == 4 {
        _mm_storeu_si32(dst_ptr as *mut _, v8);
    } else if CN == 3 {
        let pixel_3 = _mm_extract_epi8::<2>(v8);
        _mm_storeu_si16(dst_ptr, v8);
        dst_ptr.add(2).write_unaligned(pixel_3 as u8);
    } else if CN == 2 {
        _mm_storeu_si16(dst_ptr, v8);
    } else {
        let pixel_s32 = _mm_extract_epi32::<0>(v8);
        let pixel_bytes = pixel_s32.to_le_bytes();
        dst_ptr.write_unaligned(pixel_bytes[0]);
    }
}

/// Stores u32 up to x4 as u8 up to x4 based on channels count
#[inline(always)]
#[allow(dead_code)]
pub(crate) unsafe fn store_u8_s16<const CN: usize>(dst_ptr: *mut u8, regi: __m128i) {
    let v8 = _mm_packus_epi16(regi, regi);
    if CN == 4 {
        _mm_storeu_si32(dst_ptr as *mut _, v8);
    } else if CN == 3 {
        let pixel_3 = _mm_extract_epi8::<2>(v8);
        _mm_storeu_si16(dst_ptr, v8);
        dst_ptr.add(2).write_unaligned(pixel_3 as u8);
    } else if CN == 2 {
        _mm_storeu_si16(dst_ptr, v8);
    } else {
        let pixel_s32 = _mm_extract_epi32::<0>(v8);
        let pixel_bytes = pixel_s32.to_le_bytes();
        dst_ptr.write_unaligned(pixel_bytes[0]);
    }
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn store_u16_u32<const CN: usize>(dst_ptr: *mut u16, regi: __m128i) {
    let v0 = _mm_packus_epi32(regi, regi);
    if CN == 4 {
        _mm_storeu_si64(dst_ptr as *mut _, v0);
    } else if CN == 3 {
        _mm_storeu_si32(dst_ptr as *mut _, v0);
        let val = _mm_extract_epi16::<2>(v0);
        dst_ptr.add(2).write_unaligned(val as u16);
    } else if CN == 2 {
        _mm_storeu_si32(dst_ptr as *mut _, v0);
    } else {
        _mm_storeu_si16(dst_ptr as *mut _, v0);
    }
}

#[inline(always)]
#[cfg(feature = "sse")]
pub(crate) unsafe fn write_u8<const CN: usize>(dst_ptr: *mut u8, v8: __m128i) {
    if CN == 4 {
        _mm_storeu_si32(dst_ptr as *mut _, v8);
    } else if CN == 3 {
        let pixel_3 = _mm_extract_epi8::<2>(v8);
        _mm_storeu_si16(dst_ptr, v8);
        dst_ptr.add(2).write_unaligned(pixel_3 as u8);
    } else if CN == 2 {
        _mm_storeu_si16(dst_ptr, v8);
    } else {
        let pixel_s32 = _mm_extract_epi8::<0>(v8);
        dst_ptr.write_unaligned(pixel_s32 as u8);
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_hsum_ps(v: __m128) -> f32 {
    let mut shuf = _mm_movehdup_ps(v);
    let mut sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

// #[inline(always)]
// #[target_feature(enable = "sse4.1")]
// pub(crate) unsafe   fn _mm_hsum_epi32(v: __m128i) -> i32 {
//     const SHUFFLE_1: i32 = shuffle(1, 0, 3, 2);
//     let hi64 = _mm_shuffle_epi32::<SHUFFLE_1>(v);
//     let sum64 = _mm_add_epi32(hi64, v);
//     let hi32 = _mm_shufflelo_epi16::<SHUFFLE_1>(sum64); // Swap the low two elements
//     let sum32 = _mm_add_epi32(sum64, hi32);
//     _mm_cvtsi128_si32(sum32)
// }

#[inline(always)]
pub(crate) unsafe fn _mm_loadu_si128_x2(ptr: *const u8) -> (__m128i, __m128i) {
    (
        _mm_loadu_si128(ptr as *const __m128i),
        _mm_loadu_si128(ptr.add(16) as *const __m128i),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_loadu_ps_x4(ptr: *const f32) -> (__m128, __m128, __m128, __m128) {
    (
        _mm_loadu_ps(ptr),
        _mm_loadu_ps(ptr.add(4)),
        _mm_loadu_ps(ptr.add(8)),
        _mm_loadu_ps(ptr.add(12)),
    )
}

#[inline(always)]
pub(crate) unsafe fn _mm_loadu_ps_x2(ptr: *const f32) -> (__m128, __m128) {
    (_mm_loadu_ps(ptr), _mm_loadu_ps(ptr.add(4)))
}

#[inline(always)]
pub(crate) unsafe fn _mm_erase_last_ps(item: __m128) -> __m128 {
    #[allow(overflowing_literals)]
    let mask = _mm_castsi128_ps(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0));
    _mm_and_ps(item, mask)
}

#[inline(always)]
pub(crate) unsafe fn _mm_split_rgb_5_ps(
    vals: (__m128, __m128, __m128, __m128),
) -> (__m128, __m128, __m128, __m128, __m128) {
    let first = _mm_erase_last_ps(vals.0);

    let second = _mm_erase_last_ps(_mm_castsi128_ps(_mm_alignr_epi8::<12>(
        _mm_castps_si128(vals.1),
        _mm_castps_si128(vals.0),
    )));
    let third = _mm_erase_last_ps(_mm_castsi128_ps(_mm_alignr_epi8::<8>(
        _mm_castps_si128(vals.2),
        _mm_castps_si128(vals.1),
    )));
    let fourth = _mm_erase_last_ps(_mm_castsi128_ps(_mm_alignr_epi8::<4>(
        _mm_castps_si128(vals.3),
        _mm_castps_si128(vals.2),
    )));
    let fifth = _mm_erase_last_ps(vals.3);
    (first, second, third, fourth, fifth)
}

#[inline(always)]
pub(crate) unsafe fn _mm_broadcast_first(item: __m128) -> __m128 {
    const FLAG: i32 = _shuffle(0, 0, 0, 0);
    _mm_shuffle_ps::<FLAG>(item, item)
}

#[inline(always)]
pub(crate) unsafe fn _mm_broadcast_second(item: __m128) -> __m128 {
    const FLAG: i32 = _shuffle(1, 1, 1, 1);
    _mm_shuffle_ps::<FLAG>(item, item)
}

#[inline(always)]
pub(crate) unsafe fn _mm_broadcast_third(item: __m128) -> __m128 {
    const FLAG: i32 = _shuffle(2, 2, 2, 2);
    _mm_shuffle_ps::<FLAG>(item, item)
}

#[inline(always)]
pub(crate) unsafe fn _mm_broadcast_fourth(item: __m128) -> __m128 {
    const FLAG: i32 = _shuffle(3, 3, 3, 3);
    _mm_shuffle_ps::<FLAG>(item, item)
}

#[inline(always)]
pub(crate) unsafe fn load_f32_f16<const CN: usize>(ptr: *const f16) -> __m128 {
    if CN == 4 {
        let in_regi = _mm_loadu_si64(ptr as *const u8);
        return _mm_cvtph_ps(in_regi);
    } else if CN == 3 {
        let casted_ptr = ptr as *const i16;
        let in_regi = _mm_setr_epi16(
            casted_ptr.read_unaligned(),
            casted_ptr.add(1).read_unaligned(),
            casted_ptr.add(2).read_unaligned(),
            0,
            0,
            0,
            0,
            0,
        );
        return _mm_cvtph_ps(in_regi);
    } else if CN == 2 {
        let casted_ptr = ptr as *const i16;
        let in_regi = _mm_setr_epi16(
            casted_ptr.read_unaligned(),
            casted_ptr.add(1).read_unaligned(),
            0,
            0,
            0,
            0,
            0,
            0,
        );
        return _mm_cvtph_ps(in_regi);
    }
    let casted_ptr = ptr as *const i16;
    let in_regi = _mm_setr_epi16(casted_ptr.read_unaligned(), 0, 0, 0, 0, 0, 0, 0);
    _mm_cvtph_ps(in_regi)
}

#[inline(always)]
pub(crate) unsafe fn store_f32_f16<const CN: usize>(dst_ptr: *mut f16, in_regi: __m128) {
    let out_regi = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(in_regi);
    if CN == 4 {
        _mm_storeu_si64(dst_ptr as *mut u8, out_regi);
    } else if CN == 3 {
        let casted_ptr = dst_ptr as *mut i16;
        let item1 = _mm_extract_epi16::<2>(out_regi) as i16;
        _mm_storeu_si32(dst_ptr as *mut u8, out_regi);
        casted_ptr.add(2).write_unaligned(item1);
    } else if CN == 2 {
        _mm_storeu_si32(dst_ptr as *mut u8, out_regi);
    } else {
        _mm_storeu_si16(dst_ptr as *mut u8, out_regi);
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_by_3_epi32(v: __m128i) -> __m128i {
    _mm_add_epi32(_mm_slli_epi32::<1>(v), v)
}

#[inline(always)]
pub(crate) unsafe fn _mm_opt_fnmlaf_ps<const FMA: bool>(a: __m128, b: __m128, c: __m128) -> __m128 {
    if FMA {
        _mm_fnmadd_ps(b, c, a)
    } else {
        _mm_sub_ps(a, _mm_mul_ps(b, c))
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_opt_fmlaf_ps<const FMA: bool>(a: __m128, b: __m128, c: __m128) -> __m128 {
    if FMA {
        _mm_fmadd_ps(b, c, a)
    } else {
        _mm_add_ps(a, _mm_mul_ps(b, c))
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_opt_fnmlsf_ps<const FMA: bool>(a: __m128, b: __m128, c: __m128) -> __m128 {
    if FMA {
        _mm_fmsub_ps(b, c, a)
    } else {
        _mm_sub_ps(_mm_mul_ps(b, c), a)
    }
}
