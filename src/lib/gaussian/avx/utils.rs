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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub const fn _mm256_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
pub(crate) unsafe fn load_u8_u32_one(ptr: *const u8) -> __m256i {
    let u_first = u32::from_le_bytes([ptr.read_unaligned(), 0, 0, 0]);
    return _mm256_setr_epi32(u_first as i32, 0, 0, 0, 0, 0, 0, 0);
}

#[inline(always)]
pub unsafe fn avx2_pack_u32(s_1: __m256i, s_2: __m256i) -> __m256i {
    let packed = _mm256_packus_epi32(s_1, s_2);
    const MASK: i32 = _mm256_shuffle(3, 1, 2, 0);
    return _mm256_permute4x64_epi64::<MASK>(packed);
}

#[inline(always)]
pub unsafe fn avx2_pack_u16(s_1: __m256i, s_2: __m256i) -> __m256i {
    let packed = _mm256_packus_epi16(s_1, s_2);
    const MASK: i32 = _mm256_shuffle(3, 1, 2, 0);
    return _mm256_permute4x64_epi64::<MASK>(packed);
}

#[inline(always)]
pub(crate) unsafe fn load_u8_f32_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> __m256 {
    let u_first = u32::from_le_bytes([ptr.read_unaligned(), 0, 0, 0]);
    let u_second = u32::from_le_bytes([ptr.add(1).read_unaligned(), 0, 0, 0]);
    let u_third = u32::from_le_bytes([ptr.add(2).read_unaligned(), 0, 0, 0]);
    let u_fourth = match CHANNELS_COUNT {
        4 => u32::from_le_bytes([ptr.add(3).read_unaligned(), 0, 0, 0]),
        _ => 0,
    };
    let v_int = _mm256_setr_epi32(u_first as i32, u_second as i32, u_third as i32, u_fourth as i32, 0,0,0,0);
    return _mm256_cvtepi32_ps(v_int);
}