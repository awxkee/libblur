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

use half::f16;
use std::arch::asm;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn load_f32<const CHANNELS_COUNT: usize>(ptr: *const f32) -> __m128 {
    if CHANNELS_COUNT == 4 {
        return _mm_loadu_ps(ptr);
    } else if CHANNELS_COUNT == 3 {
        return _mm_setr_ps(
            ptr.read_unaligned(),
            ptr.add(1).read_unaligned(),
            ptr.add(2).read_unaligned(),
            0f32,
        );
    } else if CHANNELS_COUNT == 2 {
        return _mm_setr_ps(
            ptr.read_unaligned(),
            ptr.add(1).read_unaligned(),
            0f32,
            0f32,
        );
    }
    _mm_setr_ps(ptr.read_unaligned(), 0f32, 0f32, 0f32)
}

#[inline(always)]
pub(crate) unsafe fn load_u8_s32_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> __m128i {
    // LLVM generates a little trash code until opt-level is 3 so better here is to use assembly
    if CHANNELS_COUNT == 4 {
        let mut regi: __m128i;
        // It is preferred to allow LLVM re-use zeroed register
        let zeros = _mm_setzero_si128();
        asm!("\
           movd {1}, dword ptr [{0}]
           punpcklbw {1}, {2}
           punpcklwd {1}, {2}
    \
    ",
        in(reg) ptr,
        out(xmm_reg) regi,
        in(xmm_reg) zeros);
        regi
    } else if CHANNELS_COUNT == 3 {
        let mut regi: __m128i;
        // It is preferred to allow LLVM re-use zeroed register
        let zeros = _mm_setzero_si128();
        asm!("\
            movzx   {t1}, byte ptr [{0}]
            movzx   {t2}, word ptr [{0} + 1]
            shl {t2}, 8
            or {t1}, {t2}
            movd    {1}, {t1}
            punpcklbw {1}, {2}
            punpcklwd {1}, {2}
    \
    ",
        in(reg) ptr,
        out(xmm_reg) regi,
        in(xmm_reg) zeros,
        t1 = out(reg) _, t2 = out(reg) _);
        regi
    } else if CHANNELS_COUNT == 2 {
        _mm_setr_epi32(
            ptr.read_unaligned() as i32,
            ptr.add(1).read_unaligned() as i32,
            0,
            0,
        )
    } else {
        _mm_setr_epi32(ptr.read_unaligned() as i32, 0, 0, 0)
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_ps_epi32(ab: __m128i, cd: __m128) -> __m128i {
    let cvt = _mm_cvtepi32_ps(ab);
    const ROUNDING_FLAGS: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    let rs = _mm_round_ps::<ROUNDING_FLAGS>(_mm_mul_ps(cvt, cd));
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

pub(crate) const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
pub(crate) unsafe fn _mm_packus_epi64(a: __m128i, b: __m128i) -> __m128i {
    const SHUFFLE_MASK: i32 = shuffle(3, 1, 2, 0);
    let a = _mm_shuffle_epi32::<SHUFFLE_MASK>(a);
    let b1 = _mm_shuffle_epi32::<SHUFFLE_MASK>(b);
    _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b1)))
}

#[inline(always)]
pub(crate) unsafe fn store_f32<const CHANNELS_COUNT: usize>(dst_ptr: *mut f32, regi: __m128) {
    if CHANNELS_COUNT == 4 {
        _mm_storeu_ps(dst_ptr, regi);
    } else if CHANNELS_COUNT == 3 {
        _mm_storeu_si64(dst_ptr as *mut u8, _mm_castps_si128(regi));
        dst_ptr
            .add(2)
            .write_unaligned(f32::from_bits(_mm_extract_ps::<2>(regi) as u32));
    } else if CHANNELS_COUNT == 2 {
        _mm_storeu_si64(dst_ptr as *mut u8, _mm_castps_si128(regi));
    } else {
        _mm_storeu_si32(dst_ptr as *mut u8, _mm_castps_si128(regi));
    }
}

/// Stores u32 up to x4 as u8 up to x4 based on channels count
#[inline(always)]
#[cfg(feature = "sse")]
pub(crate) unsafe fn store_u8_s32<const CHANNELS_COUNT: usize>(dst_ptr: *mut u8, regi: __m128i) {
    let s16 = _mm_packs_epi32(regi, regi);
    let v8 = _mm_packus_epi16(s16, s16);
    if CHANNELS_COUNT == 4 {
        _mm_storeu_si32(dst_ptr, v8);
    } else if CHANNELS_COUNT == 3 {
        let pixel_3 = _mm_extract_epi8::<2>(v8);
        _mm_storeu_si16(dst_ptr, v8);
        dst_ptr.add(2).write_unaligned(pixel_3 as u8);
    } else if CHANNELS_COUNT == 2 {
        _mm_storeu_si16(dst_ptr, v8);
    } else {
        let pixel_s32 = _mm_extract_epi32::<0>(v8);
        let pixel_bytes = pixel_s32.to_le_bytes();
        dst_ptr.write_unaligned(pixel_bytes[0]);
    }
}

/// Stores u32 up to x4 as u8 up to x4 based on channels count
#[inline(always)]
pub(crate) unsafe fn store_u8_u32<const CHANNELS_COUNT: usize>(dst_ptr: *mut u8, regi: __m128i) {
    let s16 = _mm_packus_epi32(regi, regi);
    let v8 = _mm_packus_epi16(s16, s16);
    if CHANNELS_COUNT == 4 {
        _mm_storeu_si32(dst_ptr, v8);
    } else if CHANNELS_COUNT == 3 {
        let pixel_3 = _mm_extract_epi8::<2>(v8);
        _mm_storeu_si16(dst_ptr, v8);
        dst_ptr.add(2).write_unaligned(pixel_3 as u8);
    } else if CHANNELS_COUNT == 2 {
        _mm_storeu_si16(dst_ptr, v8);
    } else {
        let pixel_s32 = _mm_extract_epi32::<0>(v8);
        let pixel_bytes = pixel_s32.to_le_bytes();
        dst_ptr.write_unaligned(pixel_bytes[0]);
    }
}

#[inline(always)]
#[cfg(feature = "sse")]
pub(crate) unsafe fn write_u8<const CHANNELS_COUNT: usize>(dst_ptr: *mut u8, v8: __m128i) {
    if CHANNELS_COUNT == 4 {
        _mm_storeu_si32(dst_ptr, v8);
    } else if CHANNELS_COUNT == 3 {
        let pixel_3 = _mm_extract_epi8::<2>(v8);
        _mm_storeu_si16(dst_ptr, v8);
        dst_ptr.add(2).write_unaligned(pixel_3 as u8);
    } else if CHANNELS_COUNT == 2 {
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
    const FLAG: i32 = shuffle(0, 0, 0, 0);
    _mm_shuffle_ps::<FLAG>(item, item)
}

#[inline(always)]
pub(crate) unsafe fn _mm_broadcast_second(item: __m128) -> __m128 {
    const FLAG: i32 = shuffle(1, 1, 1, 1);
    _mm_shuffle_ps::<FLAG>(item, item)
}

#[inline(always)]
pub(crate) unsafe fn _mm_broadcast_third(item: __m128) -> __m128 {
    const FLAG: i32 = shuffle(2, 2, 2, 2);
    _mm_shuffle_ps::<FLAG>(item, item)
}

#[inline(always)]
pub(crate) unsafe fn _mm_broadcast_fourth(item: __m128) -> __m128 {
    const FLAG: i32 = shuffle(3, 3, 3, 3);
    _mm_shuffle_ps::<FLAG>(item, item)
}

#[inline(always)]
pub(crate) unsafe fn load_f32_f16<const CHANNELS_COUNT: usize>(ptr: *const f16) -> __m128 {
    if CHANNELS_COUNT == 4 {
        let in_regi = _mm_loadu_si64(ptr as *const u8);
        return _mm_cvtph_ps(in_regi);
    } else if CHANNELS_COUNT == 3 {
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
    } else if CHANNELS_COUNT == 2 {
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
pub(crate) unsafe fn store_f32_f16<const CHANNELS_COUNT: usize>(
    dst_ptr: *mut f16,
    in_regi: __m128,
) {
    let out_regi = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(in_regi);
    if CHANNELS_COUNT == 4 {
        _mm_storeu_si64(dst_ptr as *mut u8, out_regi);
    } else if CHANNELS_COUNT == 3 {
        let casted_ptr = dst_ptr as *mut i16;
        let item1 = _mm_extract_epi16::<2>(out_regi) as i16;
        _mm_storeu_si32(dst_ptr as *mut u8, out_regi);
        casted_ptr.add(2).write_unaligned(item1);
    } else if CHANNELS_COUNT == 2 {
        _mm_storeu_si32(dst_ptr as *mut u8, out_regi);
    } else {
        _mm_storeu_si16(dst_ptr as *mut u8, out_regi);
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_mul_by_3_epi32(v: __m128i) -> __m128i {
    _mm_add_epi32(_mm_slli_epi32::<1>(v), v)
}
