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

use crate::neon::f16_utils::{
    xreinterpret_f16_u16, xreinterpret_u16_f16, xvcvt_f16_f32, xvcvt_f32_f16, xvld_f16, xvst_f16,
};
use half::f16;
use std::arch::aarch64::*;
use std::arch::asm;

#[inline(always)]
pub unsafe fn load_u8_s32_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> int32x4_t {
    vreinterpretq_s32_u32(load_u8_u32_fast::<CHANNELS_COUNT>(ptr))
}

#[inline(always)]
pub unsafe fn store_f32<const CHANNELS_COUNT: usize>(dst_ptr: *mut f32, regi: float32x4_t) {
    if CHANNELS_COUNT == 4 {
        vst1q_f32(dst_ptr, regi);
    } else if CHANNELS_COUNT == 3 {
        let lo_part = vget_lane_u64::<0>(vreinterpret_u64_f32(vget_low_f32(regi)));
        (dst_ptr as *mut u64).write_unaligned(lo_part);
        dst_ptr.add(2).write_unaligned(vgetq_lane_f32::<2>(regi));
    } else if CHANNELS_COUNT == 2 {
        let lo_part = vget_lane_u64::<0>(vreinterpret_u64_f32(vget_low_f32(regi)));
        (dst_ptr as *mut u64).write_unaligned(lo_part);
    } else {
        dst_ptr.write_unaligned(vgetq_lane_f32::<0>(regi));
    }
}

#[inline(always)]
pub unsafe fn store_u8_s32<const CHANNELS_COUNT: usize>(dst_ptr: *mut u8, regi: int32x4_t) {
    let s16 = vreinterpret_u16_s16(vqmovn_s32(regi));
    let u16_f = vcombine_u16(s16, s16);
    let v8 = vqmovn_u16(u16_f);
    let pixel_u32 = vget_lane_u32::<0>(vreinterpret_u32_u8(v8));
    if CHANNELS_COUNT == 4 {
        let casted_dst = dst_ptr as *mut u32;
        casted_dst.write_unaligned(pixel_u32);
    } else if CHANNELS_COUNT == 3 {
        let bits = pixel_u32.to_le_bytes();
        let first_byte = u16::from_le_bytes([bits[0], bits[1]]);
        (dst_ptr as *mut u16).write_unaligned(first_byte);
        dst_ptr.add(2).write_unaligned(bits[2]);
    } else if CHANNELS_COUNT == 2 {
        let bits = pixel_u32.to_le_bytes();
        let first_byte = u16::from_le_bytes([bits[0], bits[1]]);
        (dst_ptr as *mut u16).write_unaligned(first_byte);
    } else {
        let bits = pixel_u32.to_le_bytes();
        dst_ptr.write_unaligned(bits[0]);
    }
}

#[inline(always)]
pub unsafe fn store_u8_u32<const CHANNELS_COUNT: usize>(dst_ptr: *mut u8, regi: uint32x4_t) {
    let s16 = vqmovn_u32(regi);
    let u16_f = vcombine_u16(s16, s16);
    let v8 = vqmovn_u16(u16_f);
    let pixel_u32 = vget_lane_u32::<0>(vreinterpret_u32_u8(v8));
    if CHANNELS_COUNT == 4 {
        let casted_dst = dst_ptr as *mut u32;
        casted_dst.write_unaligned(pixel_u32);
    } else if CHANNELS_COUNT == 3 {
        let bits = pixel_u32.to_le_bytes();
        let first_byte = u16::from_le_bytes([bits[0], bits[1]]);
        (dst_ptr as *mut u16).write_unaligned(first_byte);
        dst_ptr.add(2).write_unaligned(bits[2]);
    } else if CHANNELS_COUNT == 2 {
        let bits = pixel_u32.to_le_bytes();
        let first_byte = u16::from_le_bytes([bits[0], bits[1]]);
        (dst_ptr as *mut u16).write_unaligned(first_byte);
    } else {
        let bits = pixel_u32.to_le_bytes();
        dst_ptr.write_unaligned(bits[0]);
    }
}

#[inline(always)]
pub unsafe fn load_f32_fast<const CHANNELS_COUNT: usize>(ptr: *const f32) -> float32x4_t {
    if CHANNELS_COUNT == 4 {
        return vld1q_f32(ptr);
    } else if CHANNELS_COUNT == 3 {
        return vld1q_f32(
            [
                ptr.read_unaligned(),
                ptr.add(1).read_unaligned(),
                ptr.add(2).read_unaligned(),
                0f32,
            ]
            .as_ptr(),
        );
    } else if CHANNELS_COUNT == 2 {
        return vld1q_f32(
            [
                ptr.read_unaligned(),
                ptr.add(1).read_unaligned(),
                0f32,
                0f32,
            ]
            .as_ptr(),
        );
    }
    vld1q_f32([ptr.read_unaligned(), 0f32, 0f32, 0f32].as_ptr())
}

#[inline(always)]
pub unsafe fn load_u8_u32_fast<const CHANNELS_COUNT: usize>(ptr: *const u8) -> uint32x4_t {
    // LLVM generates a little trash code so better here is to use assembly
    let mut out_reg: uint32x4_t;
    if CHANNELS_COUNT == 4 {
        asm!("\
             ldr  {tmp1:w}, [{0}]           // Load the first byte into w1
             mov  {1:v}.s[0], {tmp1:w}      // Move w1 to the first lane of v0
             uxtl {1:v}.8h, {1:v}.8b
             uxtl {1:v}.4s, {1:v}.4h
        \
        ", in(reg) ptr, out(vreg) out_reg,
        tmp1 = out(reg) _);
        out_reg
    } else if CHANNELS_COUNT == 3 {
        asm!("\
             ldrb    {tmp1:w}, [{0}]           // Load the first byte into w1
             ldrb    {tmp2:w}, [{0}, #1]       // Load the second byte into w2
             ldrb    {tmp3:w}, [{0}, #2]       // Load the third byte into w3

             mov     {1:v}.s[0], {tmp1:w}      // Move w1 to the first lane of v0
             mov     {1:v}.s[1], {tmp2:w}      // Move w2 to the second lane of v0
             mov     {1:v}.s[2], {tmp3:w}      // Move w3 to the third lane of v0
        \
        ", in(reg) ptr, out(vreg) out_reg,
        tmp1 = out(reg) _,
        tmp2 = out(reg) _,
        tmp3 = out(reg) _);
        out_reg
    } else if CHANNELS_COUNT == 2 {
        asm!("\
             ldrb    {tmp1:w}, [{0}]           // Load the first byte into w1
             ldrb    {tmp2:w}, [{0}, #1]       // Load the second byte into w2

             mov     {1:v}.s[0], {tmp1:w}      // Move w1 to the first lane of v0
             mov     {1:v}.s[1], {tmp2:w}      // Move w2 to the second lane of v0
        \
        ", in(reg) ptr, out(vreg) out_reg,
        tmp1 = out(reg) _,
        tmp2 = out(reg) _);
        out_reg
    } else {
        asm!("\
             ldrb    {tmp1:w}, [{0}]           // Load the first byte into w1
             mov     {1:v}.s[0], {tmp1:w}      // Move w1 to the first lane of v0
        \
        ", in(reg) ptr, out(vreg) out_reg,
        tmp1 = out(reg) _);
        out_reg
    }
}

#[inline(always)]
pub(crate) unsafe fn vmulq_u32_f32(a: uint32x4_t, b: float32x4_t) -> uint32x4_t {
    let cvt = vcvtq_f32_u32(a);
    vcvtaq_u32_f32(vmulq_f32(cvt, b))
}

#[inline(always)]
pub(crate) unsafe fn vmulq_s32_f32(a: int32x4_t, b: float32x4_t) -> int32x4_t {
    let cvt = vcvtq_f32_s32(a);
    vcvtaq_s32_f32(vmulq_f32(cvt, b))
}

#[inline(always)]
pub unsafe fn load_u8_u16<const CHANNELS_COUNT: usize>(ptr: *const u8) -> uint16x4_t {
    if CHANNELS_COUNT == 4 {
        let u_first = u16::from_le_bytes([ptr.read_unaligned(), 0]);
        let u_second = u16::from_le_bytes([ptr.add(1).read_unaligned(), 0]);
        let u_third = u16::from_le_bytes([ptr.add(2).read_unaligned(), 0]);
        let u_fourth = u16::from_le_bytes([ptr.add(3).read_unaligned(), 0]);
        let store: [u16; 4] = [u_first, u_second, u_third, u_fourth];
        vld1_u16(store.as_ptr())
    } else if CHANNELS_COUNT == 3 {
        let u_first = u16::from_le_bytes([ptr.read_unaligned(), 0]);
        let u_second = u16::from_le_bytes([ptr.add(1).read_unaligned(), 0]);
        let u_third = u16::from_le_bytes([ptr.add(2).read_unaligned(), 0]);
        let u_fourth = 0;
        let store: [u16; 4] = [u_first, u_second, u_third, u_fourth];
        vld1_u16(store.as_ptr())
    } else if CHANNELS_COUNT == 2 {
        let u_first = u16::from_le_bytes([ptr.read_unaligned(), 0]);
        let u_second = u16::from_le_bytes([ptr.add(1).read_unaligned(), 0]);
        let u_third = 0;
        let u_fourth = 0;
        let store: [u16; 4] = [u_first, u_second, u_third, u_fourth];
        vld1_u16(store.as_ptr())
    } else {
        let u_first = u16::from_le_bytes([ptr.read_unaligned(), 0]);
        let store: [u16; 4] = [u_first, 0, 0, 0];
        vld1_u16(store.as_ptr())
    }
}

#[inline(always)]
pub unsafe fn prefer_vfmaq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    #[cfg(target_arch = "aarch64")]
    {
        vfmaq_f32(a, b, c)
    }
    #[cfg(target_arch = "arm")]
    {
        vmlaq_f32(a, b, c)
    }
}

#[inline(always)]
pub unsafe fn prefer_vfma_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    #[cfg(target_arch = "aarch64")]
    {
        vfma_f32(a, b, c)
    }
    #[cfg(target_arch = "arm")]
    {
        vmla_f32(a, b, c)
    }
}

#[inline(always)]
pub unsafe fn load_f32_f16<const CHANNELS_COUNT: usize>(ptr: *const f16) -> float32x4_t {
    if CHANNELS_COUNT == 4 {
        let cvt = xvld_f16(ptr);
        return xvcvt_f32_f16(cvt);
    } else if CHANNELS_COUNT == 3 {
        let recast = ptr as *const u16;
        let cvt = xreinterpret_f16_u16(vld1_u16(
            [
                recast.read_unaligned(),
                recast.add(1).read_unaligned(),
                recast.add(2).read_unaligned(),
                0,
            ]
            .as_ptr(),
        ));
        return xvcvt_f32_f16(cvt);
    } else if CHANNELS_COUNT == 2 {
        let recast = ptr as *const u16;
        let cvt = xreinterpret_f16_u16(vld1_u16(
            [
                recast.read_unaligned(),
                recast.add(1).read_unaligned(),
                0,
                0,
            ]
            .as_ptr(),
        ));
        return xvcvt_f32_f16(cvt);
    }
    let recast = ptr as *const u16;
    let cvt = xreinterpret_f16_u16(vld1_u16([recast.read_unaligned(), 0, 0, 0].as_ptr()));
    xvcvt_f32_f16(cvt)
}

#[inline(always)]
pub unsafe fn store_f32_f16<const CHANNELS_COUNT: usize>(dst_ptr: *mut f16, in_regi: float32x4_t) {
    let out_regi = xvcvt_f16_f32(in_regi);
    if CHANNELS_COUNT == 4 {
        xvst_f16(dst_ptr, out_regi);
    } else if CHANNELS_COUNT == 3 {
        let casted_out = xreinterpret_u16_f16(out_regi);
        let casted_ptr = dst_ptr as *mut u16;
        let lo_part = vreinterpret_u32_u16(casted_out);
        (casted_ptr as *mut u32).write_unaligned(vget_lane_u32::<0>(lo_part));
        casted_ptr
            .add(2)
            .write_unaligned(vget_lane_u16::<2>(casted_out));
    } else if CHANNELS_COUNT == 2 {
        let casted_out = xreinterpret_u16_f16(out_regi);
        let casted_ptr = dst_ptr as *mut u32;
        let lo_part = vreinterpret_u32_u16(casted_out);
        casted_ptr.write_unaligned(vget_lane_u32::<0>(lo_part));
    } else {
        let casted_out = xreinterpret_u16_f16(out_regi);
        let casted_ptr = dst_ptr as *mut u16;
        casted_ptr.write_unaligned(vget_lane_u16::<0>(casted_out));
    }
}

/// Stores up to 4 values from uint8x8_t
#[inline(always)]
pub(crate) unsafe fn store_u8x8_m4<const CHANNELS_COUNT: usize>(
    dst_ptr: *mut u8,
    in_regi: uint8x8_t,
) {
    let casted_u32 = vreinterpret_u32_u8(in_regi);
    let pixel = vget_lane_u32::<0>(casted_u32);

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

#[inline]
pub(crate) unsafe fn vmulq_by_3_s32(k: int32x4_t) -> int32x4_t {
    vaddq_s32(vshlq_n_s32::<1>(k), k)
}
