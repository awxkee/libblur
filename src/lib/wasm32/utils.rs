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
use std::arch::asm;
use std::arch::wasm32::*;

/// Stores up to 4 values from uint8x8_t
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn w_store_u8x8_m4<const CHANNELS_COUNT: usize>(dst_ptr: *mut u8, in_regi: v128) {
    let pixel = i32x4_extract_lane::<0>(in_regi) as u32;

    if CHANNELS_COUNT == 4 {
        (dst_ptr as *mut u32).write_unaligned(pixel);
    } else if CHANNELS_COUNT == 3 {
        let bits = pixel.to_le_bytes();
        let first_byte = u16::from_le_bytes([bits[0], bits[1]]);
        (dst_ptr as *mut u16).write_unaligned(first_byte);
        dst_ptr.add(2).write_unaligned(bits[2]);
    } else if CHANNELS_COUNT == 2 {
        let bits = pixel.to_le_bytes();
        let first_byte = u16::from_le_bytes([bits[0], bits[1]]);
        (dst_ptr as *mut u16).write_unaligned(first_byte);
    } else {
        let bits = pixel.to_le_bytes();
        dst_ptr.write_unaligned(bits[0]);
    }
}

/// Packs two u32x4 into one u16x8 using truncation
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn u32x4_pack_trunc_u16x8(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29>(a, b)
}

/// Packs two u16x8 into one u8x16 using truncation
#[inline]
#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub(crate) unsafe fn u16x8_pack_trunc_u8x16(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(a, b)
}

/// Packs two i64x2 into one i32x4 using truncation
#[inline]
#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub(crate) unsafe fn i32x4_pack_trunc_i64x2(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27>(a, b)
}

#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn load_u8_s32_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> v128 {
    // LLVM generates a little trash code until opt-level is 3 so better here is to use assembly
    if CHANNELS_COUNT == 4 {
        let mut undef = i32x4_splat(0);
        undef = i32x4_replace_lane::<0>(undef, (ptr as *const i32).read_unaligned());
        let k1 = u16x8_extend_low_u8x16(undef);
        let k2 = u32x4_extend_low_u16x8(k1);
        k2
    } else if CHANNELS_COUNT == 3 {
        let mut undef = i32x4_splat(0);
        undef = i16x8_replace_lane::<0>(undef, (ptr as *const i16).read_unaligned());
        undef = u8x16_replace_lane::<2>(undef, ptr.add(2).read_unaligned());
        let k1 = u16x8_extend_low_u8x16(undef);
        let k2 = u32x4_extend_low_u16x8(k1);
        k2
    } else if CHANNELS_COUNT == 2 {
        let mut undef = i32x4_splat(0);
        undef = i16x8_replace_lane::<0>(undef, (ptr as *const i16).read_unaligned());
        let k1 = u16x8_extend_low_u8x16(undef);
        let k2 = u32x4_extend_low_u16x8(k1);
        k2
    } else {
        let mut undef = i32x4_splat(0);
        undef = u8x16_replace_lane::<0>(undef, ptr.read_unaligned());
        let k1 = u16x8_extend_low_u8x16(undef);
        let k2 = u32x4_extend_low_u16x8(k1);
        k2
    }
}

#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn i32x4_mul_by_3(v: v128) -> v128 {
    i32x4_add(i32x4_shl(v, 1), v)
}
