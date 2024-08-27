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

use std::arch::asm;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use erydanos::{_mm_extract_epi64x, _mm_mul_epi64};
use half::f16;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct __mm128ix2(pub(crate) __m128i, pub(crate) __m128i);

#[inline]
pub unsafe fn _mm_mul_epi64x<const AVX512_DQVL: bool>(ab: __m128i, cd: __m128i) -> __m128i {
    if AVX512_DQVL {
        let mut result: __m128i;
        asm!("\
            vpmullq {0}, {1}, {2}
        \
        ", out(xmm_reg) result, in(xmm_reg) ab, in(xmm_reg) cd);
        result
    } else {
        _mm_mul_epi64(ab, cd)
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm_sub_epi64x2(a: __mm128ix2, b: __mm128ix2) -> __mm128ix2 {
    __mm128ix2(_mm_sub_epi64(a.0, b.0), _mm_sub_epi64(a.1, b.1))
}

#[inline(always)]
pub(crate) unsafe fn _mm_add_epi64x2(a: __mm128ix2, b: __mm128ix2) -> __mm128ix2 {
    __mm128ix2(_mm_add_epi64(a.0, b.0), _mm_add_epi64(a.1, b.1))
}

#[inline]
pub(crate) unsafe fn _mm_mul_n_epi64x2<const AVX512_DQVL: bool>(
    x: __mm128ix2,
    v: i64,
) -> __mm128ix2 {
    let v = _mm_set1_epi64x(v);
    let c0 = _mm_mul_epi64x::<AVX512_DQVL>(x.0, v);
    let c1 = _mm_mul_epi64x::<AVX512_DQVL>(x.1, v);
    __mm128ix2(c0, c1)
}

#[inline(always)]
pub(crate) unsafe fn _mm_store_epi64x2(ptr: *mut i64, x: __mm128ix2) {
    _mm_storeu_si128(ptr as *mut __m128i, x.0);
    _mm_storeu_si128(ptr.add(2) as *mut __m128i, x.1);
}

#[inline(always)]
pub(crate) unsafe fn _mm_set1_epi64x2(v: i64) -> __mm128ix2 {
    __mm128ix2(_mm_set1_epi64x(v), _mm_set1_epi64x(v))
}

#[inline(always)]
pub(crate) unsafe fn _mm_load_epi64x2(ptr: *const i64) -> __mm128ix2 {
    let v0 = _mm_loadu_si128(ptr as *const __m128i);
    let v1 = _mm_loadu_si128(ptr.add(2) as *const __m128i);
    __mm128ix2(v0, v1)
}

#[inline(always)]
pub(crate) unsafe fn load_u8_s64x2_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> __mm128ix2 {
    let u_first = i64::from_le_bytes([ptr.read_unaligned(), 0, 0, 0, 0, 0, 0, 0]);
    let u_second = i64::from_le_bytes([ptr.add(1).read_unaligned(), 0, 0, 0, 0, 0, 0, 0]);
    let u_third = i64::from_le_bytes([ptr.add(2).read_unaligned(), 0, 0, 0, 0, 0, 0, 0]);
    let u_fourth = match CHANNELS_COUNT {
        4 => i64::from_le_bytes([ptr.add(3).read_unaligned(), 0, 0, 0, 0, 0, 0, 0]),
        _ => 0,
    };
    __mm128ix2(
        _mm_set_epi64x(u_second, u_first),
        _mm_set_epi64x(u_fourth, u_third),
    )
}

#[inline]
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

#[inline]
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
    } else {
        let u_first = u32::from_le_bytes([ptr.read_unaligned(), 0, 0, 0]);
        let u_second = u32::from_le_bytes([ptr.add(1).read_unaligned(), 0, 0, 0]);
        let u_third = u32::from_le_bytes([ptr.add(2).read_unaligned(), 0, 0, 0]);
        let u_fourth = match CHANNELS_COUNT {
            4 => u32::from_le_bytes([ptr.add(3).read_unaligned(), 0, 0, 0]),
            _ => 0,
        };
        let store: [u32; 4] = [u_first, u_second, u_third, u_fourth];
        _mm_loadu_si128(store.as_ptr() as *const __m128i)
    }
}

#[inline]
pub(crate) unsafe fn load_u8_s16_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> __m128i {
    // LLVM generates a little trash code until opt-level is 3 so better here is to use assembly
    if CHANNELS_COUNT == 4 {
        let mut regi: __m128i;
        // It is preferred to allow LLVM re-use zeroed register
        let zeros = _mm_setzero_si128();
        asm!("\
           movd {1}, dword ptr [{0}]
           punpcklbw {1}, {2}
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
    \
    ",
        in(reg) ptr,
        out(xmm_reg) regi,
        in(xmm_reg) zeros,
        t1 = out(reg) _, t2 = out(reg) _);
        regi
    } else {
        let u_first = i16::from_le_bytes([ptr.read_unaligned(), 0]);
        let u_second = i16::from_le_bytes([ptr.add(1).read_unaligned(), 0]);
        let u_third = i16::from_le_bytes([ptr.add(2).read_unaligned(), 0]);
        let u_fourth = match CHANNELS_COUNT {
            4 => i16::from_le_bytes([ptr.add(3).read_unaligned(), 0]),
            _ => 0,
        };
        _mm_setr_epi16(u_first, u_second, u_third, u_fourth, 0, 0, 0, 0)
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
pub unsafe fn _mm_blendv_epi32x(xmm0: __m128i, xmm1: __m128i, mask: __m128i) -> __m128i {
    _mm_castps_si128(_mm_blendv_ps(
        _mm_castsi128_ps(xmm0),
        _mm_castsi128_ps(xmm1),
        _mm_castsi128_ps(mask),
    ))
}

pub const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline]
pub(crate) unsafe fn _mm_packus_epi64(a: __m128i, b: __m128i) -> __m128i {
    const SHUFFLE_MASK: i32 = shuffle(3, 1, 2, 0);
    let a = _mm_shuffle_epi32::<SHUFFLE_MASK>(a);
    let b1 = _mm_shuffle_epi32::<SHUFFLE_MASK>(b);
    let moved = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b1)));
    moved
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn load_u8_f32_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> __m128 {
    let vl = load_u8_s32_fast::<CHANNELS_COUNT>(ptr);
    _mm_cvtepi32_ps(vl)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn load_u8_u32_one(ptr: *const u8) -> __m128i {
    let u_first = u32::from_le_bytes([ptr.read_unaligned(), 0, 0, 0]);
    _mm_set1_epi32(u_first as i32)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn load_u8_u16_one(ptr: *const u8) -> __m128i {
    _mm_setr_epi16(ptr.read_unaligned() as i16, 0, 0, 0, 0, 0, 0, 0)
}

#[inline(always)]
pub(crate) unsafe fn store_f32<const CHANNELS_COUNT: usize>(dst_ptr: *mut f32, regi: __m128) {
    if CHANNELS_COUNT == 4 {
        _mm_storeu_ps(dst_ptr, regi);
    } else if CHANNELS_COUNT == 3 {
        let lo_part = _mm_extract_epi64x::<0>(_mm_castps_si128(regi));
        (dst_ptr as *mut i64).write_unaligned(lo_part);
        dst_ptr
            .add(2)
            .write_unaligned(f32::from_bits(_mm_extract_ps::<2>(regi) as u32));
    } else if CHANNELS_COUNT == 2 {
        let lo_part = _mm_extract_epi64x::<0>(_mm_castps_si128(regi));
        (dst_ptr as *mut i64).write_unaligned(lo_part);
    } else {
        dst_ptr.write_unaligned(f32::from_bits(_mm_extract_ps::<0>(regi) as u32));
    }
}

/// Stores u32 up to x4 as u8 up to x4 based on channels count
#[inline(always)]
pub(crate) unsafe fn store_u8_s32<const CHANNELS_COUNT: usize>(dst_ptr: *mut u8, regi: __m128i) {
    let s16 = _mm_packs_epi32(regi, regi);
    let v8 = _mm_packus_epi16(s16, s16);
    let pixel_s32 = _mm_extract_epi32::<0>(v8);
    if CHANNELS_COUNT == 4 {
        let casted_dst = dst_ptr as *mut i32;
        casted_dst.write_unaligned(pixel_s32);
    } else if CHANNELS_COUNT == 3 {
        let pixel_bytes = pixel_s32.to_le_bytes();
        let first_byte = u16::from_le_bytes([pixel_bytes[0], pixel_bytes[1]]);
        (dst_ptr as *mut u16).write_unaligned(first_byte);
        dst_ptr.add(2).write_unaligned(pixel_bytes[2]);
    } else if CHANNELS_COUNT == 2 {
        let pixel_bytes = pixel_s32.to_le_bytes();
        let first_byte = u16::from_le_bytes([pixel_bytes[0], pixel_bytes[1]]);
        (dst_ptr as *mut u16).write_unaligned(first_byte);
    } else {
        let pixel_bytes = pixel_s32.to_le_bytes();
        dst_ptr.write_unaligned(pixel_bytes[0]);
    }
}

/// Stores u32 up to x4 as u8 up to x4 based on channels count
#[inline(always)]
pub(crate) unsafe fn store_u8_u32<const CHANNELS_COUNT: usize>(dst_ptr: *mut u8, regi: __m128i) {
    let s16 = _mm_packus_epi32(regi, regi);
    let v8 = _mm_packus_epi16(s16, s16);
    let pixel_s32 = _mm_extract_epi32::<0>(v8);
    if CHANNELS_COUNT == 4 {
        let casted_dst = dst_ptr as *mut i32;
        casted_dst.write_unaligned(pixel_s32);
    } else if CHANNELS_COUNT == 3 {
        let pixel_bytes = pixel_s32.to_le_bytes();
        let first_byte = u16::from_le_bytes([pixel_bytes[0], pixel_bytes[1]]);
        (dst_ptr as *mut u16).write_unaligned(first_byte);
        dst_ptr.add(2).write_unaligned(pixel_bytes[2]);
    } else if CHANNELS_COUNT == 2 {
        let pixel_bytes = pixel_s32.to_le_bytes();
        let first_byte = u16::from_le_bytes([pixel_bytes[0], pixel_bytes[1]]);
        (dst_ptr as *mut u16).write_unaligned(first_byte);
    } else {
        let pixel_bytes = pixel_s32.to_le_bytes();
        dst_ptr.write_unaligned(pixel_bytes[0]);
    }
}

#[inline(always)]
pub unsafe fn _mm_hsum_ps(v: __m128) -> f32 {
    let mut shuf = _mm_movehdup_ps(v);
    let mut sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

#[inline(always)]
pub unsafe fn _mm_loadu_si128_x2(ptr: *const u8) -> (__m128i, __m128i) {
    (
        _mm_loadu_si128(ptr as *const __m128i),
        _mm_loadu_si128(ptr.add(16) as *const __m128i),
    )
}

#[inline(always)]
pub unsafe fn _mm_loadu_ps_x4(ptr: *const f32) -> (__m128, __m128, __m128, __m128) {
    (
        _mm_loadu_ps(ptr),
        _mm_loadu_ps(ptr.add(4)),
        _mm_loadu_ps(ptr.add(8)),
        _mm_loadu_ps(ptr.add(12)),
    )
}

#[inline(always)]
pub unsafe fn _mm_loadu_ps_x2(ptr: *const f32) -> (__m128, __m128) {
    (_mm_loadu_ps(ptr), _mm_loadu_ps(ptr.add(4)))
}

#[inline(always)]
pub unsafe fn _mm_erase_last_ps(item: __m128) -> __m128 {
    #[allow(overflowing_literals)]
    let mask = _mm_castsi128_ps(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0));
    _mm_and_ps(item, mask)
}

#[inline(always)]
pub unsafe fn _mm_split_rgb_5_ps(
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
pub unsafe fn _mm_broadcast_first(item: __m128) -> __m128 {
    const FLAG: i32 = shuffle(0, 0, 0, 0);
    _mm_shuffle_ps::<FLAG>(item, item)
}

#[inline(always)]
pub unsafe fn _mm_broadcast_second(item: __m128) -> __m128 {
    const FLAG: i32 = shuffle(1, 1, 1, 1);
    _mm_shuffle_ps::<FLAG>(item, item)
}

#[inline(always)]
pub unsafe fn _mm_broadcast_third(item: __m128) -> __m128 {
    const FLAG: i32 = shuffle(2, 2, 2, 2);
    _mm_shuffle_ps::<FLAG>(item, item)
}

#[inline(always)]
pub unsafe fn _mm_broadcast_fourth(item: __m128) -> __m128 {
    const FLAG: i32 = shuffle(3, 3, 3, 3);
    _mm_shuffle_ps::<FLAG>(item, item)
}

#[inline]
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

#[inline]
pub(crate) unsafe fn store_f32_f16<const CHANNELS_COUNT: usize>(
    dst_ptr: *mut f16,
    in_regi: __m128,
) {
    let out_regi = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(in_regi);
    if CHANNELS_COUNT == 4 {
        std::ptr::copy_nonoverlapping(&out_regi as *const _ as *mut u8, dst_ptr as *mut u8, 8);
    } else if CHANNELS_COUNT == 3 {
        let casted_ptr = dst_ptr as *mut i16;
        let item0 = _mm_extract_epi32::<0>(out_regi);
        let item1 = _mm_extract_epi16::<2>(out_regi) as i16;
        (casted_ptr as *mut i32).write_unaligned(item0);
        casted_ptr.add(2).write_unaligned(item1);
    } else if CHANNELS_COUNT == 2 {
        let casted_ptr = dst_ptr as *mut i16;
        let item0 = _mm_extract_epi32::<0>(out_regi);
        (casted_ptr as *mut i32).write_unaligned(item0);
    } else {
        let casted_ptr = dst_ptr as *mut i16;
        let item0 = _mm_extract_epi32::<0>(out_regi) as i16;
        casted_ptr.write_unaligned(item0);
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn _mm_mul_by_3_epi32(v: __m128i) -> __m128i {
    _mm_add_epi32(_mm_slli_epi32::<1>(v), v)
}
